"""Evaluate sd_inpaint_nose on the test split.

Generates post-op full-face predictions, then computes:
  * SSIM (whole image + mask-weighted over nose region)
  * LPIPS (whole image + mask-weighted)
  * FID (whole image distribution vs real post faces)

Saves:
  * Per-sample generated images to ``out/sd_inpaint_eval/<sid>_gen.jpg``
  * Qualitative grid to ``out/sd_inpaint_eval/qualitative_grid.png``
  * Metrics JSON to ``out/sd_inpaint_eval/metrics.json``

Usage:
    python -m ml.evaluate_sd_inpaint \\
        --base /root/CS31/models/sd_base/inpaint \\
        --lora /root/CS31/models/outcome/sd_inpaint_nose/best \\
        --out /root/CS31/out/sd_inpaint_eval
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .config import ARTIFACTS_DIR
from .data import load_pairs

logger = logging.getLogger(__name__)


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB to a float32 [0, 1] ndarray, shape (H, W, 3)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def mask_weighted_mean(per_pixel: np.ndarray, mask: np.ndarray) -> float:
    """Weighted mean of a per-pixel metric (H, W) by soft mask (H, W in [0, 1]).

    Used for SSIM where we already have the per-pixel SSIM map; for LPIPS we
    compute on masked crops instead since LPIPS doesn't expose a map.
    """
    total = float((per_pixel * mask).sum())
    weight = float(mask.sum())
    if weight < 1e-6:
        return 0.0
    return total / weight


def evaluate(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")

    # Lazy imports so the module can be inspected even if diffusers etc. aren't installed.
    from backend.inference_sd import generate_sd, load_sd_pipeline
    from skimage.metrics import structural_similarity as ssim_fn
    import lpips as lpips_mod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = load_sd_pipeline(args.base, args.lora, device=device)
    lpips_fn = lpips_mod.LPIPS(net="alex").to(device).eval()

    pairs_dir = ARTIFACTS_DIR / "dataset" / f"pairs_aligned_{args.image_size}"
    masks_dir = ARTIFACTS_DIR / "dataset" / f"masks_{args.image_size}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(exist_ok=True)

    items = load_pairs(split=args.split, limit=args.limit)
    items = [
        it for it in items
        if (pairs_dir / f"{it.sample_id}_pre.jpg").exists()
        and (pairs_dir / f"{it.sample_id}_post.jpg").exists()
        and (masks_dir / f"{it.sample_id}_mask.png").exists()
    ]
    logger.info("Evaluating %d samples on split=%s", len(items), args.split)

    # ---- Per-sample generation + pixel metrics ----
    ssim_full, ssim_nose = [], []
    lpips_full, lpips_nose = [], []
    grid_samples = []

    for it in tqdm(items, desc="generating"):
        sid = it.sample_id
        pre = Image.open(pairs_dir / f"{sid}_pre.jpg").convert("RGB")
        post = Image.open(pairs_dir / f"{sid}_post.jpg").convert("RGB")
        mask = Image.open(masks_dir / f"{sid}_mask.png").convert("L")

        gen = generate_sd(
            pipe, pre, mask,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            strength=args.strength,
            generator_seed=args.seed + hash(sid) % 10_000 if args.seed is not None else None,
            image_size=args.image_size,
        )
        gen.save(gen_dir / f"{sid}_gen.jpg", quality=95)

        # Pixel metrics (work in numpy float32 in [0, 1]).
        gen_np = pil_to_numpy(gen)
        post_np = pil_to_numpy(post)
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0  # (H, W)
        mask_np_3ch = np.stack([mask_np] * 3, axis=-1)

        # SSIM (scikit-image returns scalar OR per-pixel map via full=True)
        ssim_score, ssim_map = ssim_fn(
            post_np, gen_np, channel_axis=2, data_range=1.0, full=True,
        )
        ssim_full.append(float(ssim_score))
        # Nose-region SSIM: mean over mask. ssim_map is (H, W, 3) — avg channels.
        ssim_map_gray = ssim_map.mean(axis=-1)
        ssim_nose.append(mask_weighted_mean(ssim_map_gray, mask_np))

        # LPIPS (torch tensors in [-1, 1]).
        def to_lpips(arr):
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            return t * 2.0 - 1.0

        with torch.no_grad():
            lpips_full_val = lpips_fn(to_lpips(gen_np), to_lpips(post_np))
            lpips_full.append(float(lpips_full_val.item()))
            # Mask-region LPIPS: blend unmasked region of both to identical
            # (so only the nose contributes to perceptual distance).
            post_nose_only = post_np * mask_np_3ch + gen_np * (1 - mask_np_3ch)
            lpips_nose_val = lpips_fn(to_lpips(gen_np), to_lpips(post_nose_only))
            lpips_nose.append(float(lpips_nose_val.item()))

        if len(grid_samples) < args.grid_n:
            grid_samples.append({"pre": pre, "post": post, "gen": gen})

    # ---- FID over full images (real post vs generated) ----
    fid_score = None
    if args.compute_fid:
        fid_score = _compute_fid(gen_dir, pairs_dir, items, suffix="_post.jpg", device=device)

    metrics = {
        "split": args.split,
        "n_samples": len(items),
        "ssim_full_mean": float(np.mean(ssim_full)),
        "ssim_nose_mean": float(np.mean(ssim_nose)),
        "lpips_full_mean": float(np.mean(lpips_full)),
        "lpips_nose_mean": float(np.mean(lpips_nose)),
        "fid": fid_score,
        "config": {
            "base": str(args.base),
            "lora": str(args.lora),
            "steps": args.steps,
            "guidance": args.guidance,
            "strength": args.strength,
            "image_size": args.image_size,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("metrics: %s", json.dumps(metrics, indent=2))

    # ---- Qualitative grid ----
    if grid_samples:
        _save_grid(grid_samples, out_dir / "qualitative_grid.png")

    return metrics


def _compute_fid(gen_dir: Path, real_dir: Path, items, suffix: str, device: torch.device) -> float:
    """FID between sd_inpaint generated faces and real post-op faces.

    Uses pytorch-fid's commandline-equivalent API. Creates a temporary
    "real" subdir with only the test-split post images symlinked, so the
    real distribution matches exactly the samples we generated against.
    """
    import tempfile
    from pytorch_fid.fid_score import calculate_fid_given_paths

    with tempfile.TemporaryDirectory() as tmp:
        real_subset = Path(tmp) / "real"
        real_subset.mkdir()
        for it in items:
            src = real_dir / f"{it.sample_id}{suffix}"
            if src.exists():
                (real_subset / src.name).symlink_to(src)
        fid = calculate_fid_given_paths(
            [str(real_subset), str(gen_dir)],
            batch_size=32,
            device=device,
            dims=2048,
        )
    return float(fid)


def _save_grid(samples, path: Path) -> None:
    """Save a (N rows × 3 cols) grid of pre / real-post / generated."""
    from torchvision.utils import save_image
    import torchvision.transforms.functional as TF

    cols = []
    for s in samples:
        row = torch.cat([
            TF.to_tensor(s["pre"]),
            TF.to_tensor(s["post"]),
            TF.to_tensor(s["gen"]),
        ], dim=2)  # concat horizontally
        cols.append(row)
    grid = torch.cat(cols, dim=1)  # concat vertically
    save_image(grid, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SD Inpaint LoRA on test split")
    parser.add_argument("--base", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-n", type=int, default=10)
    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument("--no-fid", dest="compute_fid", action="store_false")
    parser.set_defaults(compute_fid=True)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
