import argparse
import csv
import logging
import tempfile
from pathlib import Path
from typing import Optional

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .config import BENCHMARK_PATH, EVAL_DIR, MODELS_DIR
from .data import PairImageDataset, denormalize
from .landmarks import LEFT_EYE_OUTER, NOSE_TIP, RIGHT_EYE_OUTER, detect_landmarks
from .runtime import checkpoint_path, get_device, load_model_from_checkpoint, model_output

logger = logging.getLogger(__name__)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = denormalize(tensor).clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def _heuristic_roi_box(width: int, height: int) -> tuple:
    """Fallback proportional ROI for profile views when landmark detection fails."""
    return (int(width * 0.3), int(height * 0.18),
            int(width * 0.75), int(height * 0.7))


def detect_roi_box(anchor_image: Image.Image) -> tuple:
    """Detect nose ROI bounding box on an anchor image (typically the reference/target).

    Returns (x1, y1, x2, y2) pixel coordinates. Uses landmark detection when
    available, falls back to a heuristic proportional crop otherwise.
    """
    result = detect_landmarks(anchor_image)
    if result.face_detected and result.nose_roi:
        return result.nose_roi
    return _heuristic_roi_box(*anchor_image.size)


def apply_roi_box(image: Image.Image, box: tuple, output_size: tuple = (96, 96)) -> Image.Image:
    """Crop `image` using a pre-computed ROI box and resize to fixed output size.

    Using the same box for pred and target keeps ROI metrics comparable and
    avoids double landmark detection per sample.
    """
    return image.crop(box).resize(output_size, Image.Resampling.LANCZOS)


def roi_crop(image: Image.Image, output_size: tuple = (96, 96)) -> Image.Image:
    """Single-image ROI crop (kept for backwards compatibility). Prefer
    detect_roi_box + apply_roi_box pair so pred and target share coordinates.
    """
    return apply_roi_box(image, detect_roi_box(image), output_size)


def _detect_facing_from_batch(pre: torch.Tensor, post: torch.Tensor) -> str:
    """Return "left", "right", or "unknown" from the first sample.

    Runs landmark detection on the first target image of the batch and
    reads the sign of (nose.x - eye_midpoint.x): positive -> nose is right
    of the eyes in image coords -> face points RIGHT; negative -> LEFT.
    Returns "unknown" when detection fails so the caller can decide the
    policy (e.g. fall back to the dataset's dominant orientation) rather
    than silently defaulting to "right" as this function used to.

    KNOWN LIMITATION: facing is detected on the FIRST item only and applied
    to the whole batch. If a batch mixes left- and right-facing profiles
    the ROI window will be wrong for most samples. In practice the dataset
    is canonicalized to right-facing in prepare_pairs (and batch_size is
    typically 4), so within-batch facing is usually consistent; the
    batch-averaged LPIPS score absorbs the occasional off-sample. If this
    assumption ever breaks, move detection inside the per-sample loop.
    """
    try:
        anchor = tensor_to_image(post[0].cpu())
        result = detect_landmarks(anchor)
        if not result.face_detected or not result.landmarks:
            return "unknown"
        # Use the shared landmark-index constants so this stays consistent
        # with the rest of the codebase instead of magic numbers.
        max_idx = max(NOSE_TIP, RIGHT_EYE_OUTER, LEFT_EYE_OUTER)
        if len(result.landmarks) <= max_idx:
            return "unknown"
        nose_x = result.landmarks[NOSE_TIP][0]
        eye_mid_x = (
            result.landmarks[RIGHT_EYE_OUTER][0] + result.landmarks[LEFT_EYE_OUTER][0]
        ) / 2.0
        return "right" if (nose_x - eye_mid_x) >= 0 else "left"
    except Exception:
        return "unknown"


def _get_roi_bounds(shape, sample_ids, pre: torch.Tensor, post: torch.Tensor):
    """Return (y1, y2, x1, x2) ROI bounds for tensor-level metrics.

    The legacy bounds (25%-75% horizontally, 18%-70% vertically) assume the
    subject faces right. For left-facing profiles the nose is on the left
    side of the frame; mirror x-bounds horizontally so the crop still
    covers the nose. When facing is unknown we default to "right" (matches
    the dataset's dominant orientation) but log the fall-back so poor
    detection rates surface instead of silently skewing metrics.
    """
    _, _, h_t, w_t = shape
    # Right-facing default: nose lives in the right half of the frame, so the
    # ROI window is biased right-of-center (0.30-0.85 horizontally). The
    # previous bounds (0.25-0.75) were symmetric about the vertical midpoint,
    # so mirroring for left-facing profiles returned the SAME window and the
    # whole facing-detection plumbing above was a no-op. Asymmetric bounds
    # make left/right mirror produce meaningfully different crops.
    y1 = int(h_t * 0.20)
    y2 = int(h_t * 0.75)
    x1 = int(w_t * 0.30)
    x2 = int(w_t * 0.85)
    facing = _detect_facing_from_batch(pre, post)
    if facing == "unknown":
        logger.warning(
            "Facing undetected for ROI bounds; defaulting to 'right'. "
            "Persistent warnings indicate landmark-detection regressions."
        )
        facing = "right"
    if facing == "left":
        # Mirror horizontal bounds: (0.30, 0.85) -> (0.15, 0.70) about the
        # vertical midpoint, putting the window on the left half for
        # left-facing profiles.
        x1, x2 = w_t - x2, w_t - x1
    return y1, y2, x1, x2


def compute_ssim(pred: Image.Image, target: Image.Image) -> float:
    # np.asarray is ~100x faster than list(getdata()) + tensor roundtrip
    pred_arr = np.asarray(pred.convert("RGB"), dtype=np.float32)
    target_arr = np.asarray(target.convert("RGB"), dtype=np.float32)
    return float(structural_similarity(pred_arr, target_arr, channel_axis=2, data_range=255))


def save_grid(rows: list[tuple[Image.Image, Image.Image, Image.Image]], destination: Path) -> None:
    columns = 3
    figure, axes = plt.subplots(len(rows), columns, figsize=(9, 3 * max(1, len(rows))))
    if len(rows) == 1:
        axes = [axes]
    titles = ["Pre-op", "Real Post-op", "Generated Post-op"]
    for row_index, row in enumerate(rows):
        for col_index, image in enumerate(row):
            axes[row_index][col_index].imshow(image)
            axes[row_index][col_index].axis("off")
            if row_index == 0:
                axes[row_index][col_index].set_title(titles[col_index])
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def append_benchmark_row(payload: dict) -> None:
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = BENCHMARK_PATH.exists()
    with BENCHMARK_PATH.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "model",
                "checkpoint",
                "sample_count",
                "fid_dims",
                "ssim",
                "roi_ssim",
                "lpips",
                "roi_lpips",
                "fid",
                # Device matters for cross-run comparability: LPIPS on MPS
                # vs CPU can drift by ~1e-3 depending on backend precision.
                "device",
                "metric_device",
            ],
            # Legacy rows don't have device/metric_device; tolerate extras
            # so we don't crash reading an existing CSV.
            extrasaction="ignore",
        )
        if not exists:
            writer.writeheader()
        writer.writerow(payload)


def write_manual_review_template(model_name: str, sample_ids: list[str]) -> None:
    path = EVAL_DIR / f"{model_name}_manual_review_template.csv"
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "sample_id",
                "visual_credibility",
                "nasal_naturalness",
                "artifact_severity",
                "notes",
            ],
        )
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "visual_credibility": "",
                    "nasal_naturalness": "",
                    "artifact_severity": "",
                    "notes": "",
                }
            )


def evaluate_model(model_name: str, limit: Optional[int], checkpoint_name: str, fid_dims: int) -> dict:
    device = get_device()
    dataset = PairImageDataset(split="test", limit=limit)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    model, _ = load_model_from_checkpoint(model_name, checkpoint_name=checkpoint_name, device=device)

    # LPIPS: on MPS, AlexNet weights hit the same torchvision compatibility
    # issues as pytorch-fid's Inception. Keep LPIPS on CPU for MPS so it
    # and FID are both on CPU (apples-to-apples across metric backends,
    # recorded in the benchmark CSV so drifts are visible).
    metric_device = device if device.type == "cuda" else torch.device("cpu")
    metric_lpips = lpips.LPIPS(net="alex").to(metric_device)
    model_dir = EVAL_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    preview_rows = []
    sample_ids = []
    ssim_scores = []
    roi_ssim_scores = []
    lpips_scores = []
    roi_lpips_scores = []

    with tempfile.TemporaryDirectory() as temp_dir:
        fake_dir = Path(temp_dir) / "fake"
        real_dir = Path(temp_dir) / "real"
        fake_dir.mkdir(parents=True, exist_ok=True)
        real_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {model_name}"):
                pre = batch["pre"].to(device)
                post = batch["post"].to(device)
                generated = model_output(model_name, model, pre)

                # Move generated/post to the metric device (CPU on MPS so
                # we don't double-pay for a half-supported kernel path).
                gen_metric = generated.to(metric_device)
                post_metric = post.to(metric_device)
                lp = metric_lpips(gen_metric, post_metric).mean().item()
                # Proportional ROI crop for tensor-level LPIPS. For per-image
                # left/right-facing adaptation see _get_roi_bounds; we detect
                # orientation via landmarks and mirror x-bounds when the face
                # points left so the crop actually includes the nose.
                y1_t, y2_t, x1_t, x2_t = _get_roi_bounds(
                    generated.shape, batch.get("sample_id"), pre, post,
                )
                roi_generated = gen_metric[:, :, y1_t:y2_t, x1_t:x2_t]
                roi_target = post_metric[:, :, y1_t:y2_t, x1_t:x2_t]
                roi_lp = metric_lpips(roi_generated, roi_target).mean().item()
                lpips_scores.append(lp)
                roi_lpips_scores.append(roi_lp)

                for index in range(pre.shape[0]):
                    sid = batch["sample_id"][index]
                    sample_ids.append(sid)
                    pred_image = tensor_to_image(generated[index].cpu())
                    target_image = tensor_to_image(post[index].cpu())
                    pre_image = tensor_to_image(pre[index].cpu())
                    save_image(denormalize(generated[index]), fake_dir / f"{sid}.png")
                    save_image(denormalize(post[index]), real_dir / f"{sid}.png")
                    ssim_scores.append(compute_ssim(pred_image, target_image))
                    # Use the TARGET image as anchor for ROI box, then apply the
                    # same coordinates to the prediction. This keeps roi_ssim
                    # comparable across samples and avoids pred-side detection
                    # failing on distorted generations.
                    roi_box = detect_roi_box(target_image)
                    ssim_roi = compute_ssim(
                        apply_roi_box(pred_image, roi_box),
                        apply_roi_box(target_image, roi_box),
                    )
                    roi_ssim_scores.append(ssim_roi)
                    if len(preview_rows) < 8:
                        preview_rows.append((pre_image, target_image, pred_image))

        # pytorch-fid accepts torch-style device strings. Fall back to CPU on MPS
        # because pytorch-fid's Inception weights are not fully MPS-compatible
        # in all torchvision versions.
        fid_device = "cuda" if device.type == "cuda" else "cpu"
        fid_value = fid_score.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=4,
            device=fid_device,
            dims=fid_dims,
        )

    grid_path = model_dir / "qualitative_grid.png"
    save_grid(preview_rows, grid_path)
    write_manual_review_template(model_name, sample_ids[:50])

    result = {
        "model": model_name,
        "checkpoint": str(checkpoint_path(model_name, checkpoint_name)),
        "sample_count": len(sample_ids),
        "fid_dims": fid_dims,
        "ssim": sum(ssim_scores) / max(1, len(ssim_scores)),
        "roi_ssim": sum(roi_ssim_scores) / max(1, len(roi_ssim_scores)),
        "lpips": sum(lpips_scores) / max(1, len(lpips_scores)),
        "roi_lpips": sum(roi_lpips_scores) / max(1, len(roi_lpips_scores)),
        "fid": float(fid_value),
        "device": str(device),
        "metric_device": str(metric_device),
    }
    append_benchmark_row(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained outcome models.")
    parser.add_argument("--model", default="all", help="Model to evaluate or 'all'.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--fid-dims", type=int, default=2048)
    args = parser.parse_args()

    if args.model == "all":
        model_dirs = [path.name for path in (MODELS_DIR / "outcome").glob("*") if path.is_dir()]
        model_names = sorted(model_dirs)
    else:
        model_names = [args.model]

    results = []
    for model_name in model_names:
        results.append(evaluate_model(model_name, args.limit, args.checkpoint, args.fid_dims))

    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
