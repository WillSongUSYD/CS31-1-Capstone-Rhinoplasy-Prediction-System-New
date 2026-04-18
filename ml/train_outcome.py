import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .config import ARTIFACTS_DIR, DEFAULT_IMAGE_SIZE, PAIR_256_DIR, ensure_directories
from .data import NoseROIDataset, PairImageDataset, denormalize
from .models.common import set_requires_grad
from .runtime import (
    _base_model_name, create_model, get_device, model_dir, model_output,
    save_checkpoint, save_metadata,
)


def validation_l1(model_name: str, model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.L1Loss()
    losses = []
    with torch.no_grad():
        for batch in loader:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            generated = model_output(model_name, model, pre)
            losses.append(float(criterion(generated, post).detach().cpu()))
    return sum(losses) / max(1, len(losses))


def save_preview(model_name: str, model, loader: DataLoader, device: torch.device) -> None:
    preview_dir = ARTIFACTS_DIR / "eval" / model_name
    preview_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader))
    pre = batch["pre"].to(device)
    post = batch["post"].to(device)
    with torch.no_grad():
        generated = model_output(model_name, model, pre)
    for index in range(min(6, pre.shape[0])):
        save_image(denormalize(pre[index]), preview_dir / f"{batch['sample_id'][index]}_pre.png")
        save_image(denormalize(post[index]), preview_dir / f"{batch['sample_id'][index]}_post.png")
        save_image(denormalize(generated[index]), preview_dir / f"{batch['sample_id'][index]}_pred.png")


def _dataloader_kwargs(device: torch.device, shuffle: bool) -> dict:
    """Return DataLoader kwargs tuned for the target device."""
    # On MPS/CUDA, worker processes speed up image decoding and augmentation.
    # On CPU we leave workers at 0 to avoid overhead.
    use_workers = device.type in {"cuda", "mps"}
    # Bumped from 4/2 to 8/4 for 48GB M-series machines where the old defaults
    # left the GPU starved during 512x512 batches.
    return {
        "num_workers": 8 if use_workers and shuffle else (4 if use_workers else 0),
        "persistent_workers": use_workers,
        "pin_memory": device.type == "cuda",
        "shuffle": shuffle,
    }


def train(model_name: str, epochs: int, batch_size: int, lr: float, limit: Optional[int], image_size: int, nose_only: bool = False) -> Path:
    ensure_directories()
    if not PAIR_256_DIR.exists():
        raise FileNotFoundError("Prepared pairs not found. Run python -m ml.prepare_pairs first.")

    device = get_device()
    if nose_only:
        train_dataset = NoseROIDataset(split="train", limit=limit, image_size=image_size, augment=True)
        val_dataset = NoseROIDataset(split="val", limit=max(4, limit // 4) if limit else None, image_size=image_size)
    else:
        train_dataset = PairImageDataset(split="train", limit=limit, image_size=image_size, augment=True)
        val_dataset = PairImageDataset(split="val", limit=max(4, limit // 4) if limit else None, image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, **_dataloader_kwargs(device, shuffle=True))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, **_dataloader_kwargs(device, shuffle=False))

    model = create_model(model_name).to(device)
    base_name = _base_model_name(model_name)
    history = []
    best_val = float("inf")

    if base_name == "autoencoder":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    elif base_name == "pix2pix":
        optimizer_g = Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_d = Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    elif base_name == "cyclegan":
        optimizer_g = Adam(list(model.g_xy.parameters()) + list(model.g_yx.parameters()), lr=lr, betas=(0.5, 0.999))
        optimizer_d = Adam(list(model.d_x.parameters()) + list(model.d_y.parameters()), lr=lr, betas=(0.5, 0.999))
    elif base_name == "diffusion":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{epochs}")
        for batch in progress:
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)

            if base_name == "autoencoder":
                optimizer.zero_grad()
                loss, losses = model.step(pre, post)
                loss.backward()
                optimizer.step()
                running_loss += losses.reconstruction
                progress.set_postfix(loss=losses.reconstruction)

            elif base_name == "pix2pix":
                try:
                    # Generate once; detach for D step, re-use graph for G step.
                    fake = model(pre)

                    # ===== Discriminator step =====
                    set_requires_grad([model.discriminator], True)
                    optimizer_d.zero_grad()
                    d_loss = model.discriminator_loss(pre, post, fake.detach())
                    d_loss.backward()
                    optimizer_d.step()

                    # ===== Generator step (D frozen so only G receives gradient) =====
                    set_requires_grad([model.discriminator], False)
                    optimizer_g.zero_grad()
                    g_loss, adv_loss, l1_loss = model.generator_loss(pre, post, fake)
                    g_loss.backward()
                    optimizer_g.step()
                finally:
                    # Always leave D with requires_grad=True for the next iteration
                    set_requires_grad([model.discriminator], True)
                running_loss += g_loss.detach().item()
                progress.set_postfix(g_loss=g_loss.detach().item(), d_loss=d_loss.detach().item())

            elif base_name == "cyclegan":
                try:
                    # ===== Generator step (freeze discriminators) =====
                    set_requires_grad([model.d_x, model.d_y], False)
                    optimizer_g.zero_grad()
                    g_total, losses, fake_y_det, fake_x_det = model.generator_loss(pre, post)
                    g_total.backward()
                    optimizer_g.step()

                    # ===== Discriminator step (freeze generators) =====
                    # Reuse the detached fakes from the G step instead of
                    # re-running the generators - saves two full forward passes.
                    set_requires_grad([model.d_x, model.d_y], True)
                    set_requires_grad([model.g_xy, model.g_yx], False)
                    optimizer_d.zero_grad()
                    d_total = model.discriminator_loss(pre, post, fake_y=fake_y_det, fake_x=fake_x_det)
                    d_total.backward()
                    optimizer_d.step()
                finally:
                    # Restore both networks to trainable state regardless of
                    # which step raised, so a transient error in one iteration
                    # doesn't permanently leave requires_grad=False.
                    set_requires_grad([model.d_x, model.d_y], True)
                    set_requires_grad([model.g_xy, model.g_yx], True)

                running_loss += losses.generator_total
                progress.set_postfix(g_loss=losses.generator_total, d_loss=d_total.detach().item())

            elif base_name == "diffusion":
                optimizer.zero_grad()
                loss, losses = model(pre, post)
                loss.backward()
                optimizer.step()
                running_loss += losses.noise_loss
                progress.set_postfix(loss=losses.noise_loss)

        avg_train = running_loss / max(1, len(train_loader))
        val_score = validation_l1(model_name, model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": avg_train, "val_l1": val_score})

        checkpoint_payload = {
            "model_name": model_name,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "history": history,
            "image_size": image_size,
        }
        save_checkpoint(model_name, checkpoint_payload, name="latest.pt")
        if val_score < best_val:
            best_val = val_score
            save_checkpoint(model_name, checkpoint_payload, name="best.pt")

    save_metadata(
        model_name,
        {
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "limit": limit,
            "image_size": image_size,
            "device": str(device),
            "history": history,
            "best_val_l1": best_val,
        },
    )
    save_preview(model_name, model, val_loader, device)
    history_path = model_dir(model_name) / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model_dir(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an outcome prediction model.")
    parser.add_argument("--model", required=True, choices=["autoencoder", "pix2pix", "cyclegan", "diffusion"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None,
                        help=f"Image size. Default: 256 for full-face, 128 for --nose-only.")
    parser.add_argument("--nose-only", action="store_true", help="Train on nose ROI crops only")
    args = parser.parse_args()
    effective_model = args.model
    if args.nose_only:
        effective_model = f"{args.model}_nose"
        if args.image_size is None:
            args.image_size = 128  # default for nose ROI
    elif args.image_size is None:
        args.image_size = DEFAULT_IMAGE_SIZE
    directory = train(
        model_name=effective_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit=args.limit,
        image_size=args.image_size,
        nose_only=args.nose_only,
    )
    print(f"Saved model artifacts to {directory}")


if __name__ == "__main__":
    main()
