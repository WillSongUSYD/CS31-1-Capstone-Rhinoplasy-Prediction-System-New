"""SD 1.5 Inpainting + LoRA inference adapter for the backend serve layer.

Usage pattern mirrors ``inference.py``:

    from backend.inference_sd import load_sd_pipeline, generate_sd

    pipe = load_sd_pipeline(
        base_dir="models/sd_base/inpaint",
        lora_dir="models/outcome/sd_inpaint_nose/best",
    )
    result_image = generate_sd(pipe, pre_image_pil, nose_mask_pil)

The pipeline is cached by ``(base_dir, lora_dir, device)`` so repeated
requests don't re-read 4GB of weights from disk.

Kept separate from ``inference.py`` rather than bolted on because:
  * SD pipeline has a different interface (PIL in/PIL out, scheduler, guidance)
  * Loads 4GB; the existing LRU cap of 4 models doesn't account for that
  * Loading diffusers lazily avoids pulling the dependency when the server
    only serves cyclegan/autoencoder/pix2pix.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# Separate cache from inference.py's cache: SD pipelines are ~4GB each,
# we cap at 2 to stay under typical 8-16GB VRAM envelopes even with the
# base torch-backed models loaded alongside.
_SD_CACHE_MAX = 2
_sd_cache: "OrderedDict[Tuple[str, str, str], object]" = OrderedDict()
_sd_cache_lock = threading.Lock()


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS has intermittent numerical issues with SD attention; fall back to
    # CPU on Apple silicon is too slow (~minutes per 512x512 image). We
    # prefer to fail loudly than silently return 10-minute latencies.
    if torch.backends.mps.is_available():
        logger.warning(
            "MPS detected for SD inference. Attention numerics are unstable on "
            "some MPS builds; if output is garbled, set CUDA_VISIBLE_DEVICES and "
            "use GPU, or force CPU via device='cpu' (very slow)."
        )
        return torch.device("mps")
    return torch.device("cpu")


def load_sd_pipeline(
    base_dir: str | Path,
    lora_dir: str | Path,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
):
    """Load (or reuse cached) SD Inpainting pipeline with LoRA attached.

    Raises FileNotFoundError if either the base or LoRA directory is missing,
    so callers get a clear error rather than a cryptic HF Hub 404.
    """
    base_dir = Path(base_dir).resolve()
    lora_dir = Path(lora_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"SD base model not found at {base_dir}")
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA adapter not found at {lora_dir}")

    device = device or _get_device()
    # float16 has no native CPU compute path — every op is emulated by
    # up-casting to float32, computing, then down-casting, which makes CPU
    # inference dramatically slower than plain float32. float16 only pays
    # off on CUDA. Force float32 whenever we run on CPU (Windows / Intel
    # Mac), so a prediction takes minutes rather than tens of minutes.
    if device.type == "cpu" and dtype == torch.float16:
        logger.info(
            "CPU device — using float32 (float16 inference is far slower on CPU)"
        )
        dtype = torch.float32
    key = (str(base_dir), str(lora_dir), str(device))

    with _sd_cache_lock:
        if key in _sd_cache:
            _sd_cache.move_to_end(key)
            return _sd_cache[key]

    # Release lock while we load (4GB read + move-to-device takes 10+ s).
    # A concurrent request for the same pipeline will load twice; acceptable
    # in practice because SD requests are rare + expensive.
    logger.info("loading SD inpaint pipeline base=%s lora=%s device=%s dtype=%s",
                base_dir.name, lora_dir.name, device, dtype)
    _load_t0 = time.perf_counter()
    from ml.models.sd_inpaint import build_inference_pipeline
    pipe = build_inference_pipeline(base_dir, lora_dir, device=device, dtype=dtype)
    # Enable memory-efficient attention (if xformers is installed). This is
    # a no-op if xformers isn't available; diffusers logs a helpful warning.
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except (ImportError, ModuleNotFoundError, AttributeError):
        # xformers not installed or diffusers version too old - fall back
        # to native SDPA (PyTorch 2.x) which is still fast on Blackwell.
        pass
    # channels-last memory layout speeds up the conv-heavy UNet/VAE on CPU
    # (oneDNN) and on GPU. Harmless if the backend doesn't support it.
    try:
        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    except Exception:  # pragma: no cover
        pass
    # Disable progress bar for clean logs during serving.
    pipe.set_progress_bar_config(disable=True)
    logger.info("SD pipeline loaded in %.1fs", time.perf_counter() - _load_t0)

    with _sd_cache_lock:
        _sd_cache[key] = pipe
        _sd_cache.move_to_end(key)
        # LRU evict
        while len(_sd_cache) > _SD_CACHE_MAX:
            evict_key, evict_pipe = _sd_cache.popitem(last=False)
            logger.info("evicting SD pipeline %s from cache", evict_key)
            del evict_pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pipe


def generate_sd(
    pipeline,
    pre_image: Image.Image,
    mask_image: Image.Image,
    prompt: str = "a post-rhinoplasty face, refined natural nose, clear skin, photorealistic",
    negative_prompt: str = "blurry, distorted, cartoon, low quality, deformed",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 1.0,
    generator_seed: Optional[int] = None,
    image_size: int = 512,
) -> Image.Image:
    """Run inpainting. Returns a PIL.Image of the generated post-op face.

    Args
    ----
    pre_image : pre-op aligned face (will be resized to ``image_size``).
    mask_image : grayscale nose mask (255 = regenerate, 0 = preserve).
        Will be resized to ``image_size``.
    num_inference_steps : 20-50 is the useful range. 30 balances quality vs
        latency (~3 s/image on 5090).
    guidance_scale : SD 1.5's default is 7.5. Higher → more prompt-adherent
        but less natural. For our fixed prompt and LoRA-fine-tuned base,
        7.5 works well.
    strength : 1.0 regenerates the masked region from pure Gaussian noise,
        matching our LoRA training distribution (t=T). Values <1.0 on a
        9-channel inpainting pipeline produce partial blending that the
        LoRA wasn't trained for. Kept as a parameter only for A/B
        experiments - leave at 1.0 for correct behaviour.
    """
    # Standardise inputs. Diffusers pipelines handle PIL internally; do the
    # resize here for a predictable output size regardless of source file.
    if pre_image.mode != "RGB":
        pre_image = pre_image.convert("RGB")
    if mask_image.mode != "L":
        mask_image = mask_image.convert("L")
    if pre_image.size != (image_size, image_size):
        pre_image = pre_image.resize((image_size, image_size), Image.LANCZOS)
    if mask_image.size != (image_size, image_size):
        mask_image = mask_image.resize((image_size, image_size), Image.BILINEAR)

    generator = None
    if generator_seed is not None:
        device = pipeline.unet.device
        generator = torch.Generator(device=device).manual_seed(int(generator_seed))

    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pre_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
            height=image_size,
            width=image_size,
        )
    return result.images[0]
