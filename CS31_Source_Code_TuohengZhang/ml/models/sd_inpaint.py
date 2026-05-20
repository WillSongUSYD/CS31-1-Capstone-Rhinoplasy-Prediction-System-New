"""Stable Diffusion 1.5 Inpainting + LoRA fine-tune for rhinoplasty outcome.

The task: given a pre-op aligned face (512x512) and a soft-edge nose mask
(512x512 grayscale), predict the post-op aligned face. SD Inpainting is a
natural fit because the mask semantically says "regenerate the nose region,
keep everything else."

We fine-tune via LoRA on the UNet's attention layers only. VAE and text
encoder are frozen. Trainable params ~= 2-3M (vs 860M full UNet).

This module exposes three things:
  * ``load_pipeline_components(base_path, device, dtype)`` - loads VAE,
    text encoder+tokenizer, UNet, scheduler separately. For training.
  * ``attach_lora_to_unet(unet, rank, alpha)`` - injects LoRA adapters
    targeting to_q/to_k/to_v/to_out.0 of attention blocks. Returns a
    PEFT-wrapped model whose save_pretrained() writes only adapter weights.
  * ``build_inference_pipeline(base_path, lora_dir, device, dtype)`` -
    loads a ``StableDiffusionInpaintPipeline`` with LoRA weights attached.
    For ``backend/inference_sd.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


# Fixed text prompt during training and inference. We don't have per-sample
# captions in the rhinoplasty dataset, so a short semantic anchor biases the
# base model toward the post-op distribution instead of drifting into SD's
# general face prior.
DEFAULT_PROMPT = (
    "a post-rhinoplasty face, refined natural nose, clear skin, photorealistic"
)

# LoRA targets: every attention projection inside UNet cross+self-attn blocks.
# We exclude MLP layers to keep param count low; attention is where SD learns
# "which regions correspond to which tokens", exactly what a mask-conditioned
# task needs to re-wire.
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

# Text encoder LoRA targets (CLIP's naming differs from UNet's).
# CLIP uses HuggingFace transformers naming: q/k/v/out_proj on attention.
LORA_TEXT_ENCODER_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


def attach_lora_to_text_encoder(
    text_encoder,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
):
    """Attach LoRA adapters to a CLIP text encoder for joint training.

    For pure image generation fine-tuning (style LoRAs) the text encoder
    is usually frozen. For our task - where a fixed prompt carries strong
    semantic intent ("post-rhinoplasty face") - training a small
    (rank=8) text encoder LoRA lets the prompt token embeddings drift
    toward the target distribution, effectively learning a
    prompt-specific embedding in situ.

    Uses transformers' own native ``add_adapter`` path via the same
    LoraConfig used on UNet, just with CLIP-specific target module names.
    """
    text_encoder.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=LORA_TEXT_ENCODER_TARGETS,
        lora_dropout=dropout,
        bias="none",
    )
    text_encoder.add_adapter(config)
    trainable = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError(
            "Text encoder LoRA attach produced 0 trainable params. "
            "Check transformers version and LORA_TEXT_ENCODER_TARGETS."
        )
    logger.info("Text encoder LoRA attached: %d trainable params", trainable)
    return text_encoder


def load_pipeline_components(
    base_path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[AutoencoderKL, CLIPTextModel, CLIPTokenizer, UNet2DConditionModel, DDPMScheduler]:
    """Load VAE / text encoder / tokenizer / UNet / scheduler individually.

    Returns them ready for a training loop: VAE and text encoder with
    ``requires_grad_(False)``, UNet in float32 (LoRA attaches on top).

    ``base_path`` is the local HF cache directory for a SD 1.5 Inpainting
    checkpoint (9-channel UNet). Loading separately (rather than via
    ``StableDiffusionInpaintPipeline.from_pretrained``) avoids instantiating
    the inference-only safety checker and gives us handles to each module
    for the train step.
    """
    base_path = str(base_path)
    # VAE and text encoder: use caller-supplied dtype (typically fp16 in
    # training to save VRAM - they're frozen so no precision issues).
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae").to(device, dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder").to(device, dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer")
    # UNet: ALWAYS load in fp32. LoRA fine-tuning needs fp32 master weights
    # for the adapter matrices to train stably; autocast handles the fp16
    # forward for speed. Ignoring the caller's dtype here on purpose.
    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet").to(device, dtype=torch.float32)
    scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    # Freeze VAE and text encoder. We only train the UNet (via LoRA).
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # Sanity: the inpainting UNet has 9 input channels (4 latent + 1 mask +
    # 4 masked-image-latent). A regular SD 1.5 checkpoint has 4. If someone
    # accidentally downloaded the wrong base, we catch it here instead of
    # deep in the train loop with a cryptic shape mismatch.
    in_channels = unet.config.in_channels
    if in_channels != 9:
        raise ValueError(
            f"Expected SD 1.5 Inpainting UNet (9 input channels), got {in_channels}. "
            f"Check that {base_path!r} is a *-inpainting checkpoint, not plain SD 1.5."
        )

    return vae, text_encoder, tokenizer, unet, scheduler


def attach_lora_to_unet(
    unet: UNet2DConditionModel,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
) -> UNet2DConditionModel:
    """Attach LoRA adapters to the UNet in-place via diffusers' native API.

    Uses ``unet.add_adapter(LoraConfig)`` instead of ``get_peft_model``. The
    difference matters at save/load time: add_adapter leaves the UNet class
    intact (still a ``UNet2DConditionModel``) and hooks the adapter into its
    ``state_dict()``, which means:

      * the saved adapter weights use diffusers' expected key schema
        ("unet.down_blocks.0...lora_A.weight") and load cleanly via
        ``pipeline.load_lora_weights(dir)``;
      * ``get_peft_model`` would return a ``PeftModel`` wrapper whose
        state_dict uses PEFT's own key schema ("base_model.model.down_blocks..."),
        which diffusers 0.26-0.30 load via a compat shim that has broken
        at various points. Native path is safer.

    Non-adapter params are frozen. Returns the SAME unet instance (modified
    in place) for symmetry with other model-builder helpers in this repo.

    Typical settings for SD LoRA fine-tune:
      * rank 4-16 for style, 16-64 for task adaptation
      * alpha ≈ 2 × rank (effective lr ~= lr / (alpha/rank) = lr/2)

    For our 458-sample dataset with a meaningful task shift (pre-op →
    post-op nose), rank=16/alpha=32 is a reasonable default.
    """
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
    )
    # Freeze everything first, then add_adapter sets adapter params trainable.
    unet.requires_grad_(False)
    unet.add_adapter(config)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    # Guard against silent no-op: a diffusers/peft version skew could leave
    # every param frozen and we'd burn hours of GPU time training nothing.
    # Fail loud instead.
    if trainable == 0:
        raise RuntimeError(
            "LoRA attach produced 0 trainable params. Check diffusers >=0.25 "
            "and that LORA_TARGET_MODULES matches the UNet attention module names."
        )
    logger.info(
        "LoRA attached: %d trainable / %d total (%.3f%%)",
        trainable, total, 100.0 * trainable / max(1, total),
    )
    return unet


def encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    batch_size: int,
    device: torch.device,
    require_grad: bool = False,
) -> torch.Tensor:
    """Tokenize + encode a single prompt, broadcast to batch_size.

    ``require_grad``: when True, gradients flow through the text encoder
    (needed when text encoder LoRA is trainable). When False, we wrap in
    ``no_grad`` to save activation memory (text encoder frozen).
    """
    tokens = tokenizer(
        [prompt] * batch_size,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    if require_grad:
        out = text_encoder(input_ids)
        embeddings = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    else:
        with torch.no_grad():
            # Use named attribute rather than `[0]` - transformers 5.x changed
            # the ModelOutput type; indexing isn't guaranteed to give
            # last_hidden_state.
            out = text_encoder(input_ids)
            embeddings = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return embeddings


def build_masked_latents(
    vae: AutoencoderKL,
    pre_image: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the two conditioning tensors SD Inpainting's UNet expects.

    Returns:
      * ``masked_image_latents`` [B, 4, H/8, W/8] - VAE-encoded pre-op image
        with the mask region zeroed in pixel space first (standard SD
        inpainting preprocessing).
      * ``mask_latent`` [B, 1, H/8, W/8] - nearest-neighbour downsampled mask.

    The caller concatenates these with the noisy target latent along dim=1
    to form the 9-channel UNet input.

    ``pre_image``: [B, 3, H, W] in [-1, 1] range (standard SD normalisation).
    ``mask``: [B, 1, H, W] in [0, 1] where 1.0 = "regenerate here".
    """
    # Zero the masked region in the pre image. Match diffusers reference
    # ``StableDiffusionInpaintPipeline.prepare_mask_and_masked_image`` which
    # uses a HARD threshold at 0.5 for the image multiply (so the "hole" has
    # a crisp boundary in pixel space). Our saved masks have a soft gaussian
    # σ=18 ramp - if we pass the soft mask to both the multiply AND the
    # mask_latent, inference (which hard-thresholds) sees a different input
    # distribution than training. Keep mask_latent soft (the UNet was
    # pretrained on soft conditioning), hard-threshold only the multiply.
    hard_mask = (mask > 0.5).to(mask.dtype)
    masked_pre = pre_image * (1.0 - hard_mask)

    with torch.no_grad():
        # VAE scaling factor = 0.18215 for SD 1.5; using the config value
        # keeps us robust to any future base model that tweaks it.
        masked_image_latents = vae.encode(masked_pre).latent_dist.sample()
        masked_image_latents = masked_image_latents * vae.config.scaling_factor

    # Downsample mask to latent resolution. Our masks are soft-edge
    # (gaussian σ=18 in pixel space), so we use area averaging to preserve
    # the ramp. The UNet's mask channel was trained on 0/1 binary masks in
    # the base HF checkpoint - keep range [0, 1] at latent resolution.
    h_lat = masked_image_latents.shape[-2]
    w_lat = masked_image_latents.shape[-1]
    mask_latent = torch.nn.functional.interpolate(
        mask, size=(h_lat, w_lat), mode="area"
    )

    return masked_image_latents, mask_latent


def build_inference_pipeline(
    base_path: str | Path,
    lora_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionInpaintPipeline:
    """Assemble an end-to-end inference pipeline with LoRA weights attached.

    Used by the backend serve layer and the eval harness. Returns a
    ``StableDiffusionInpaintPipeline`` ready to call. The safety checker is
    disabled (medical imagery triggers false positives on pre/post skin).
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(base_path),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # Swap default DDPM/PNDM to DPMSolver++ for inference. Same prediction
    # target (epsilon), but the trajectory solver reaches equivalent quality
    # in ~2x fewer sampling steps. Free quality-vs-latency win.
    from diffusers import DPMSolverMultistepScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2,
    )

    # Load LoRA weights — UNet keys only.
    #
    # The text encoder adapter was saved with the `lora_linear_layer.down/up`
    # naming from an older diffusers build.  Newer diffusers' get_peft_kwargs()
    # expects `lora_A/lora_B` naming; when it can't find those keys, rank_dict
    # stays empty and list(rank_dict.values())[0] raises IndexError.
    #
    # The UNet LoRA uses kohya-style `lora.down/up` naming, which the bundled
    # diffusers handles correctly.  When the text_encoder slice of the state
    # dict is absent, _load_lora_into_text_encoder's early-exit guard
    # (`if len(state_dict) > 0`) skips loading cleanly.
    _lora_file = Path(lora_dir) / "pytorch_lora_weights.safetensors"
    if not _lora_file.exists():
        raise FileNotFoundError(f"LoRA weights not found at {_lora_file}")
    from safetensors.torch import load_file as _load_safetensors
    _full_sd = _load_safetensors(str(_lora_file))
    _unet_sd = {k: v for k, v in _full_sd.items() if k.startswith("unet.")}
    if not _unet_sd:
        raise ValueError(f"No UNet LoRA keys found in {_lora_file}")
    pipe.load_lora_weights(_unet_sd)

    # Optional: fuse LoRA into base weights for ~5% inference speedup.
    # Disabled by default - keeps the pipeline reloadable with different
    # LoRA weights if the caller wants to swap adapters.

    return pipe
