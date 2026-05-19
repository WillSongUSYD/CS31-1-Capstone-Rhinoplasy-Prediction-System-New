"""Fine-tune SD 1.5 Inpainting with LoRA for rhinoplasty outcome prediction.

Dataset: 572 aligned-face pairs (458 train / 57 val / 57 test). Each sample
is ``(pre_face, nose_mask, post_face)``. We train the UNet to predict the
noise added to the post-op face latents, conditioned on:
  * the pre-op face (as masked_image_latents in the 9-channel input)
  * the nose mask (as mask_latent in the 9-channel input)
  * a fixed text prompt

Only LoRA adapters on UNet attention are trainable. VAE + text encoder
frozen. ~2-3M trainable params vs 860M full UNet.

Usage (on AutoDL 5090):
    python -m ml.train_sd_inpaint \\
        --base /root/CS31/models/sd_base/inpaint \\
        --out /root/CS31/models/outcome/sd_inpaint_nose \\
        --steps 5000 --batch-size 2 --grad-accum 2 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .config import ARTIFACTS_DIR
from .data import load_pairs
from .models.sd_inpaint import (
    DEFAULT_PROMPT,
    attach_lora_to_text_encoder,
    attach_lora_to_unet,
    build_masked_latents,
    encode_prompt,
    load_pipeline_components,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AlignedFaceInpaintDataset(Dataset):
    """Loads ``(pre_face, nose_mask, post_face)`` triplets at 512x512.

    Expects these directories under ``artifacts/dataset/``:
      * ``pairs_aligned_512/{sid}_pre.jpg``
      * ``pairs_aligned_512/{sid}_post.jpg``
      * ``masks_512/{sid}_mask.png``

    The dataset filters out samples missing any of the three files and
    raises if the filtered set is empty (matches NoseROIDataset behaviour).

    ``augment=True`` applies synchronized horizontal flip to all three
    tensors (pre, post, mask flip together). No color jitter - SD is
    sensitive to input colour statistics and the base model's VAE was
    trained on a specific normalization.
    """

    def __init__(
        self,
        split: str,
        limit: Optional[int] = None,
        image_size: int = 512,
        augment: bool = False,
    ):
        self.image_size = image_size
        self.pairs_dir = ARTIFACTS_DIR / "dataset" / f"pairs_aligned_{image_size}"
        self.masks_dir = ARTIFACTS_DIR / "dataset" / f"masks_{image_size}"

        for d in (self.pairs_dir, self.masks_dir):
            if not d.exists():
                raise FileNotFoundError(
                    f"Required directory missing: {d}. "
                    f"Generate aligned pairs + masks at size {image_size} first."
                )

        items = load_pairs(split=split, limit=limit)
        # Filter to entries where all three files actually exist on disk.
        filtered = []
        for it in items:
            sid = it.sample_id
            pre = self.pairs_dir / f"{sid}_pre.jpg"
            post = self.pairs_dir / f"{sid}_post.jpg"
            mask = self.masks_dir / f"{sid}_mask.png"
            if pre.exists() and post.exists() and mask.exists():
                filtered.append(it)
        dropped = len(items) - len(filtered)
        if dropped:
            logger.warning(
                "Filtered %d of %d %s items (split=%s) due to missing pre/post/mask files",
                dropped, len(items), self.__class__.__name__, split,
            )
        if not filtered:
            raise RuntimeError(
                f"{self.__class__.__name__} for split={split!r} is empty. "
                f"Ensure pairs_aligned_{image_size}/ and masks_{image_size}/ are populated."
            )
        self.items = filtered
        self.augment = augment and split == "train"

        # SD expects images in [-1, 1] (standard normalisation: mean=0.5, std=0.5).
        # The VAE encoder was trained under this assumption; feeding [0, 1] will
        # silently produce a colour-shifted output.
        self.image_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [1, H, W] in [0, 1]
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        import random
        it = self.items[index]
        sid = it.sample_id
        pre = Image.open(self.pairs_dir / f"{sid}_pre.jpg").convert("RGB")
        post = Image.open(self.pairs_dir / f"{sid}_post.jpg").convert("RGB")
        mask = Image.open(self.masks_dir / f"{sid}_mask.png").convert("L")

        if self.augment and random.random() > 0.5:
            # Mirrored nose mask is still a valid nose mask (our masks are
            # already oriented-right by pipeline, but flipping pre+post+mask
            # together preserves pairing regardless of original orientation).
            pre = TF.hflip(pre)
            post = TF.hflip(post)
            mask = TF.hflip(mask)

        return {
            "sample_id": sid,
            "pre": self.image_tf(pre),
            "post": self.image_tf(post),
            "mask": self.mask_tf(mask),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger.info("args = %s", vars(args))

    # Determinism: seed everything before we touch torch/numpy. Without this,
    # a crashed run cannot be resumed with identical timestep/noise sampling,
    # and we can't disentangle noise from true regressions in ablations.
    import random as _random
    import numpy as _np
    _random.seed(args.seed)
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Deterministic-but-fast cuDNN: don't lock into determinstic algorithms
    # (would hurt throughput ~20%), just seed cudnn's own RNG.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available; training on %s will be extremely slow", device)
    # TF32 matmul: free ~30% speedup on Ampere+/Blackwell for fp32 matmul
    # paths (anywhere autocast doesn't reach, e.g. optimizer state update).
    # Safe for training - TF32 keeps fp32 exponent range and ~10-bit mantissa.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- Load base model components ----
    logger.info("Loading SD 1.5 Inpainting components from %s", args.base)
    # VAE is always frozen - load in fp16 to save VRAM (no gradient path).
    # Text encoder dtype depends on whether we train LoRA on it: fp16 when
    # frozen (saves VRAM), fp32 when trained (PEFT adds fp32 LoRA adapters
    # on top - mixing fp16 base weights with fp32 LoRA adapters under
    # bf16 autocast produces NaN within a few steps).
    # UNet always fp32 master weights for LoRA training; autocast handles
    # the bf16 forward.
    te_dtype = torch.float32 if args.train_text_encoder else torch.float16
    vae, text_encoder, tokenizer, unet, scheduler = load_pipeline_components(
        args.base, device=device, dtype=torch.float16,
    )
    # load_pipeline_components loads both VAE and text_encoder in fp16
    # regardless of caller dtype for UNet. For the train-TE path we need
    # to upcast the text encoder here.
    if args.train_text_encoder:
        text_encoder = text_encoder.to(dtype=torch.float32)

    # ---- Attach LoRA ----
    unet = attach_lora_to_unet(unet, rank=args.lora_rank, alpha=args.lora_alpha)
    unet.train()

    # Optionally attach LoRA to text encoder too. For tasks with a fixed
    # semantic prompt ("post-rhinoplasty face"), a small text encoder LoRA
    # can shift the prompt embedding toward our dataset's latent space,
    # improving how UNet cross-attention responds to the prompt.
    train_text_encoder = args.train_text_encoder
    if train_text_encoder:
        text_encoder = attach_lora_to_text_encoder(
            text_encoder,
            rank=args.te_lora_rank,
            alpha=args.te_lora_alpha,
        )
        text_encoder.train()
    else:
        # Fully frozen - keep text encoder in eval mode + no_grad during forward.
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    # ---- Resume LoRA weights from a previous run ----
    # Used for chaining training runs past the initial --steps cap. Loads
    # adapter weights from a previous checkpoint dir, leaving the optimizer
    # and LR scheduler fresh (explicitly lose momentum - keeps semantics
    # simple: each chained run is a fresh optimisation on warm-started
    # weights, which behaves like "more epochs at a new LR").
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-from dir not found: {resume_path}")
        logger.info("Resuming LoRA weights from %s", resume_path)
        # Use diffusers' LoRA loader utility on the UNet directly.
        # StableDiffusionInpaintPipeline.save_lora_weights writes a
        # pytorch_lora_weights.safetensors containing both UNet and optional
        # text-encoder adapter weights keyed by prefix. We load the whole
        # state dict and dispatch to each model.
        from safetensors.torch import load_file
        lora_file = resume_path / "pytorch_lora_weights.safetensors"
        if not lora_file.exists():
            raise FileNotFoundError(
                f"pytorch_lora_weights.safetensors not found in {resume_path}")
        full_state = load_file(str(lora_file))
        # Keys are prefixed with "unet." or "text_encoder." by
        # StableDiffusionInpaintPipeline.save_lora_weights.
        unet_sd = {
            k[len("unet."):]: v for k, v in full_state.items()
            if k.startswith("unet.")
        }
        te_sd = {
            k[len("text_encoder."):]: v for k, v in full_state.items()
            if k.startswith("text_encoder.")
        }
        # Convert back from diffusers key schema to PEFT schema so
        # load_state_dict(strict=False) finds the adapter params.
        from diffusers.utils import convert_state_dict_to_peft
        if unet_sd:
            peft_unet_sd = convert_state_dict_to_peft(unet_sd)
            missing, unexpected = unet.load_state_dict(peft_unet_sd, strict=False)
            logger.info("Resumed UNet LoRA: %d keys loaded, %d missing, %d unexpected",
                        len(peft_unet_sd), len(missing), len(unexpected))
        if te_sd and train_text_encoder:
            peft_te_sd = convert_state_dict_to_peft(te_sd)
            missing, unexpected = text_encoder.load_state_dict(peft_te_sd, strict=False)
            logger.info("Resumed TE LoRA: %d keys loaded, %d missing, %d unexpected",
                        len(peft_te_sd), len(missing), len(unexpected))
        elif te_sd and not train_text_encoder:
            logger.warning("Checkpoint has text-encoder LoRA weights but "
                           "--train-text-encoder is off; skipping TE weights.")

    # ---- Optimiser ----
    # Gather trainable params from BOTH models (UNet LoRA always, text
    # encoder LoRA only when enabled). PEFT handles requires_grad filtering.
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if train_text_encoder:
        trainable_params += [p for p in text_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # ---- LR scheduler: linear warmup + cosine decay ----
    # Constant LR is fine for short runs but over 20k steps we benefit from
    # warmup (stabilises early LoRA init) and late-training lr decay
    # (refines without oscillating around the minimum).
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        # num_training_steps controls the cosine period; use total
        # optimizer steps (= steps / grad_accum).
        num_training_steps=args.steps // args.grad_accum,
    )

    # ---- Data ----
    train_ds = AlignedFaceInpaintDataset(
        split="train", limit=args.limit, image_size=args.image_size, augment=True,
    )
    val_ds = AlignedFaceInpaintDataset(
        split="val",
        limit=max(4, args.limit // 4) if args.limit else None,
        image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.workers // 2), pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    logger.info("train=%d val=%d batch=%d grad_accum=%d effective_batch=%d",
                len(train_ds), len(val_ds), args.batch_size, args.grad_accum,
                args.batch_size * args.grad_accum)

    # ---- Pre-compute prompt embeddings ----
    # When text encoder is FROZEN, the prompt embedding is fixed for the
    # whole run - compute once, reuse every step.
    # When text encoder is TRAINED (LoRA adapters added), the embedding
    # changes as adapter weights update, so we must re-encode every step.
    if not train_text_encoder:
        prompt_embeds = encode_prompt(
            tokenizer, text_encoder, args.prompt, batch_size=args.batch_size, device=device,
            require_grad=False,
        )
        logger.info("prompt=%r embeds=%s (cached - TE frozen)", args.prompt, tuple(prompt_embeds.shape))
    else:
        prompt_embeds = None
        logger.info("prompt=%r (recomputed every step - TE trainable)", args.prompt)

    # ---- Training loop ----
    num_train_timesteps = scheduler.config.num_train_timesteps
    vae_scale = vae.config.scaling_factor

    step = 0
    history = []
    best_val = float("inf")
    start_time = time.time()

    # Using steps instead of epochs because diffusion training usually
    # converges by step count; epochs on 458 samples @ batch 2 = 229 steps
    # which isn't enough to meaningfully compare. 5000 steps @ batch 2 =
    # ~22 epochs, plenty for LoRA.
    pbar = tqdm(total=args.steps, desc="sd-inpaint LoRA")
    data_iter = iter(train_loader)
    optimizer.zero_grad()

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reshuffle by recreating the iterator. Persistent workers stay
            # up so this is cheap.
            data_iter = iter(train_loader)
            batch = next(data_iter)

        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        bsz = pre.shape[0]

        # If text encoder is trained, re-encode every step (gradient flow).
        # Otherwise reuse the cached prompt embedding for identical batch
        # sizes; fall back to re-encoding on size mismatch.
        if train_text_encoder:
            current_prompt = encode_prompt(
                tokenizer, text_encoder, args.prompt, batch_size=bsz, device=device,
                require_grad=True,
            )
        elif bsz != prompt_embeds.shape[0]:
            current_prompt = encode_prompt(
                tokenizer, text_encoder, args.prompt, batch_size=bsz, device=device,
                require_grad=False,
            )
        else:
            current_prompt = prompt_embeds

        # Autocast: bf16 forward, fp32 master weights. bf16 has fp32's
        # exponent range so gradients don't underflow, which lets us skip
        # GradScaler entirely. Blackwell (sm_120) has native bf16 tensor
        # cores so throughput is identical to fp16 in practice.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            # 1. Encode target (post-op) into latents.
            with torch.no_grad():
                target_latents = vae.encode(post).latent_dist.sample() * vae_scale

            # 2. Sample noise + timestep, add noise to target latents.
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0, num_train_timesteps, (bsz,), device=device, dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)

            # 3. Build mask + masked pre-op latents for conditioning.
            masked_image_latents, mask_latent = build_masked_latents(vae, pre, mask)

            # 4. Concat: [noisy_target_latents (4ch), mask_latent (1ch),
            #             masked_image_latents (4ch)] = 9ch input.
            unet_input = torch.cat([noisy_latents, mask_latent, masked_image_latents], dim=1)

            # 5. Predict noise residual.
            model_pred = unet(
                unet_input,
                timesteps,
                encoder_hidden_states=current_prompt,
                return_dict=False,
            )[0]

            # 6. Loss (MSE between predicted and actual noise). This is
            # the standard v-prediction-free DDPM loss for epsilon-parameterised
            # schedulers.
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / args.grad_accum

        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            # Clip before stepping to prevent LoRA from taking a huge leap
            # on a rare noisy batch (common failure mode in small-dataset
            # diffusion fine-tune).
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()  # advance cosine schedule after each optim step
            optimizer.zero_grad()

        loss_val = float(loss.detach().item()) * args.grad_accum
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss_val:.4f}", step=step)

        # Periodic validation + checkpoint. We validate on the raw noise
        # prediction loss (not generation quality) because full sampling is
        # too slow to run every N steps. Visual quality is checked after
        # training via evaluate_sd_inpaint.py.
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            val_loss = _validate(
                unet, vae, val_loader, scheduler, device, prompt_embeds,
                num_train_timesteps, vae_scale, args.prompt, tokenizer, text_encoder,
                train_text_encoder=train_text_encoder,
            )
            elapsed = time.time() - start_time
            history.append({
                "step": step + 1,
                "train_loss": loss_val,
                "val_loss": val_loss,
                "elapsed_sec": round(elapsed, 1),
            })
            logger.info("[step %d] train=%.4f val=%.4f elapsed=%.1fs",
                        step + 1, loss_val, val_loss, elapsed)

            # Save "latest" every val cycle. Saves only LoRA adapter
            # weights (few MB).
            _save_lora(unet, text_encoder, out_dir / "latest",
                       step=step + 1, val_loss=val_loss,
                       train_text_encoder=train_text_encoder)

            if math.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                _save_lora(unet, text_encoder, out_dir / "best",
                           step=step + 1, val_loss=val_loss,
                           train_text_encoder=train_text_encoder)
                logger.info("  ✓ new best val_loss=%.4f saved to best/", val_loss)

            # Milestone checkpoint: save an immutable copy every
            # `milestone_every` steps so we can eval multiple points post-hoc
            # and pick the one with best FID (which may not be the lowest
            # val_loss due to overfit/LoRA-capacity interactions).
            if args.milestone_every > 0 and (step + 1) % args.milestone_every == 0:
                milestone_dir = out_dir / f"step_{step + 1}"
                _save_lora(unet, text_encoder, milestone_dir,
                           step=step + 1, val_loss=val_loss,
                           train_text_encoder=train_text_encoder)
                logger.info("  ✓ milestone checkpoint saved at step %d", step + 1)

            (out_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8",
            )

        step += 1

    pbar.close()
    # Final metadata
    (out_dir / "metadata.json").write_text(json.dumps({
        "base_model": str(args.base),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "train_text_encoder": train_text_encoder,
        "te_lora_rank": args.te_lora_rank if train_text_encoder else None,
        "te_lora_alpha": args.te_lora_alpha if train_text_encoder else None,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "milestone_every": args.milestone_every,
        "prompt": args.prompt,
        "best_val_loss": best_val,
        "total_time_sec": round(time.time() - start_time, 1),
        "history": history,
    }, indent=2), encoding="utf-8")
    logger.info("training done. best val_loss=%.4f. artifacts -> %s",
                best_val, out_dir)
    return out_dir


@torch.no_grad()
def _validate(
    unet, vae, loader, scheduler, device, prompt_embeds,
    num_train_timesteps, vae_scale, prompt_text, tokenizer, text_encoder,
    train_text_encoder: bool = False,
) -> float:
    """Mean noise-prediction MSE on the val split.

    This is NOT generation quality — it's just "how well does UNet predict
    noise for random timesteps on held-out (pre, post) pairs?". Cheaper than
    sampling but correlates with downstream FID once training is underway.
    """
    was_training = unet.training
    unet.eval()
    te_was_training = text_encoder.training if train_text_encoder else False
    if train_text_encoder:
        text_encoder.eval()
    total = 0.0
    n = 0
    for batch in loader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        mask = batch["mask"].to(device)
        bsz = pre.shape[0]

        # When text encoder is trained, always re-encode (its weights shift
        # each val cycle). When frozen, reuse cached prompt_embeds unless
        # batch size differs.
        if train_text_encoder or prompt_embeds is None:
            pe = encode_prompt(tokenizer, text_encoder, prompt_text, batch_size=bsz, device=device)
        elif bsz != prompt_embeds.shape[0]:
            pe = encode_prompt(tokenizer, text_encoder, prompt_text, batch_size=bsz, device=device)
        else:
            pe = prompt_embeds

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            target_latents = vae.encode(post).latent_dist.sample() * vae_scale
            # Deterministic noise per (sample_id, batch_index) via a seeded
            # generator so re-running validation on the same checkpoint gives
            # identical numbers. Without this, val_loss trembles step-to-step
            # and we can't tell a 2% regression from measurement noise.
            gen = torch.Generator(device=device).manual_seed(31 + n)
            noise = torch.randn(target_latents.shape, generator=gen, device=device, dtype=target_latents.dtype)
            # Average over a deterministic timestep grid instead of a single
            # fixed t. Noise-prediction difficulty varies ~10x across the
            # schedule; sampling t~Uniform on a seeded grid gives a more
            # faithful validation signal without the random walk.
            timesteps = torch.linspace(
                50, num_train_timesteps - 50, bsz, device=device,
            ).round().long()
            noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)
            masked_image_latents, mask_latent = build_masked_latents(vae, pre, mask)
            unet_input = torch.cat([noisy_latents, mask_latent, masked_image_latents], dim=1)
            pred = unet(unet_input, timesteps, encoder_hidden_states=pe, return_dict=False)[0]
            loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
        total += float(loss.item()) * bsz
        n += bsz

    if was_training:
        unet.train()
    if te_was_training:
        text_encoder.train()
    return total / max(1, n)


def _save_lora(unet, text_encoder, target: Path, step: int, val_loss: float,
               train_text_encoder: bool = False) -> None:
    """Save LoRA adapter weights in diffusers-native format.

    Uses ``StableDiffusionInpaintPipeline.save_lora_weights`` which writes
    ``pytorch_lora_weights.safetensors`` with the key schema that
    ``pipeline.load_lora_weights`` can read back without the PEFT compat
    shim. When ``train_text_encoder`` is True the text encoder adapters
    are saved alongside UNet adapters in the same file, and diffusers'
    pipeline loader will attach both when reloading.

    Adapter file is ~12-25MB depending on rank; cheap to keep many.
    """
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    target.mkdir(parents=True, exist_ok=True)

    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet)
    )
    text_encoder_lora_state_dict = None
    if train_text_encoder:
        text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(text_encoder)
        )

    StableDiffusionInpaintPipeline.save_lora_weights(
        save_directory=str(target),
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_lora_state_dict,
        safe_serialization=True,
    )
    (target / "checkpoint_meta.json").write_text(json.dumps({
        "step": step,
        "val_loss": val_loss,
        "train_text_encoder": train_text_encoder,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune SD 1.5 Inpainting on rhinoplasty pairs")
    parser.add_argument("--base", required=True, help="Path to local SD 1.5 Inpainting HF dir")
    parser.add_argument("--out", required=True, help="Output directory for LoRA + logs")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps. effective_batch = batch-size * grad-accum")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16, help="UNet LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="UNet LoRA alpha")
    parser.add_argument("--train-text-encoder", action="store_true",
                        help="Also attach LoRA to text encoder and train it")
    parser.add_argument("--te-lora-rank", type=int, default=8, help="Text encoder LoRA rank")
    parser.add_argument("--te-lora-alpha", type=int, default=16, help="Text encoder LoRA alpha")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Linear LR warmup steps before cosine decay")
    parser.add_argument("--milestone-every", type=int, default=0,
                        help="Save a step_N/ checkpoint every this many steps "
                             "(in addition to latest/ and best/). 0 = disabled.")
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31,
                        help="Seeds Python/Numpy/Torch RNG for reproducibility. "
                             "Noise sampling inside val loop has its own seeded generator.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a LoRA checkpoint directory (e.g. ./latest/) "
                             "to load adapter weights from. Use to chain multiple "
                             "training runs past the initial --steps ceiling once "
                             "milestone FID eval shows the model is still improving. "
                             "Note: only LoRA weights are restored, not optimizer "
                             "state - LR schedule starts fresh. Pair with a lower "
                             "--lr (e.g. 5e-5) to avoid undoing progress.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: train on only first N samples")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
