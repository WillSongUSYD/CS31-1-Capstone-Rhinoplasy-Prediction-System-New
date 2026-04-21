"""Runtime GPU health check.

SD 1.5 attention on some MPS builds produces NaN/Inf output or severely
distorted imagery (see existing warning in backend/inference_sd.py:44-57).
We proactively test this at app startup with a tiny 64×64 / 4-step
inpaint canary; a healthy MPS gives a numerically finite result in ~2s,
a broken one returns NaNs or garbled pixels immediately.

If the canary fails once, we persist ``force_cpu=True`` in the user
config so we don't re-probe every launch. User can reset via menu.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def probe_device(base_dir: Path, lora_dir: Path) -> tuple[str, str]:
    """Return ``(chosen_device, banner_msg_or_empty)``.

    ``base_dir`` / ``lora_dir`` are the SD Inpainting base + our LoRA.
    If either is missing we skip the probe and return ``("unknown", "")``
    — onboarding hasn't run yet and there's nothing to probe.

    The returned device string is one of ``"mps"``, ``"cuda"``, or
    ``"cpu"``. The banner is an empty string on the happy path and a
    user-facing warning on CPU fallback.
    """
    if not base_dir.exists() or not lora_dir.exists():
        return "unknown", ""

    # Deferred heavy imports.
    import numpy as np
    import torch
    from PIL import Image

    if not torch.backends.mps.is_available():
        # MPS not even present (Intel Mac or virtualised). No canary
        # needed; CPU is the only option and we shouldn't warn as if
        # something broke.
        if torch.cuda.is_available():
            return "cuda", ""
        return "cpu", ""

    try:
        from backend.inference_sd import load_sd_pipeline, generate_sd
        pipeline = load_sd_pipeline(base_dir, lora_dir)
    except Exception as exc:
        logger.warning("canary skipped: pipeline load failed: %s", exc)
        return "cpu", ("MPS 管道加载失败,已切 CPU 模式(每次生成 3-5 分钟)")

    # 64×64 canary — 4 steps @ guidance 1.0 for speed.
    # The pipeline internally enforces divisible-by-8 image_size so we
    # go with 64 which is the minimum it accepts.
    fake = Image.new("RGB", (64, 64), (128, 128, 128))
    mask = Image.new("L", (64, 64), 255)
    t0 = time.time()
    try:
        out = generate_sd(
            pipeline, fake, mask,
            prompt="canary", negative_prompt="",
            num_inference_steps=4, guidance_scale=1.0, strength=1.0,
            generator_seed=31, image_size=64,
        )
    except Exception as exc:
        logger.warning("MPS canary threw: %s — falling back to CPU", exc)
        return "cpu", "MPS 生成异常,已切 CPU 模式(每次生成 3-5 分钟)"
    dt = time.time() - t0

    arr = np.asarray(out, dtype=np.float32) / 255.0
    if not np.isfinite(arr).all():
        logger.warning("MPS canary produced NaN/Inf — falling back to CPU")
        return "cpu", "MPS 数值不稳定,已切 CPU 模式(每次生成 3-5 分钟)"

    # Sanity: a healthy pipeline produces non-degenerate output. If std
    # is effectively zero (constant-colour output), something's wrong.
    if arr.std() < 1e-3:
        logger.warning("MPS canary output is nearly constant — fallback to CPU")
        return "cpu", "MPS 输出异常,已切 CPU 模式(每次生成 3-5 分钟)"

    logger.info("MPS canary passed in %.2fs; using MPS", dt)
    return "mps", ""
