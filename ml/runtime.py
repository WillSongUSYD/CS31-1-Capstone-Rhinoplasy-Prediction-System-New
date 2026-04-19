import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .config import MODELS_DIR
from .models import AutoencoderModel, CycleGANModel, DiffusionFeasibilityModel, Pix2PixModel

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _base_model_name(model_name: str) -> str:
    """Strip '_nose' suffix to get the base architecture name."""
    name = model_name.lower()
    if name.endswith("_nose"):
        return name[:-5]
    return name


def create_model(model_name: str):
    name = _base_model_name(model_name)
    if name == "autoencoder":
        return AutoencoderModel()
    if name == "pix2pix":
        return Pix2PixModel()
    if name == "cyclegan":
        return CycleGANModel()
    if name == "diffusion":
        return DiffusionFeasibilityModel()
    raise ValueError(f"Unsupported model: {model_name}")


def model_output(model_name: str, model, pre: torch.Tensor) -> torch.Tensor:
    base = _base_model_name(model_name)
    if base == "diffusion":
        output = model.sample(pre)
        if output.shape[-1] != pre.shape[-1]:
            output = F.interpolate(output, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        return output
    return model(pre)


def model_dir(model_name: str) -> Path:
    return MODELS_DIR / "outcome" / model_name


def checkpoint_path(model_name: str, name: str = "latest.pt") -> Path:
    return model_dir(model_name) / name


def save_checkpoint(model_name: str, payload: dict, name: str = "latest.pt") -> Path:
    """Save a checkpoint as a state_dict only (weights_only-safe format).

    Writes non-tensor metadata (training history, epoch, image_size) to a
    sibling JSON sidecar keyed by checkpoint name. This avoids the pickle-RCE
    surface you hit when loading a dict-wrapped checkpoint with
    ``weights_only=False``.

    Accepts our legacy payload shape ``{"state_dict": ..., "history": ..., ...}``
    for backwards-compatibility with callers; the state_dict is extracted and
    saved independently and the rest becomes sidecar JSON.
    """
    directory = model_dir(model_name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name

    # Accept either a raw state_dict (mapping of tensors) or our legacy
    # {"state_dict": ..., "history": ...} wrapper. We detect the wrapper by
    # the presence of the "state_dict" key.
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        sidecar = {k: v for k, v in payload.items() if k != "state_dict"}
    else:
        state_dict = payload
        sidecar = {}

    # Convert state_dict tensors to CPU for portable checkpoint files. Saving
    # while tensors still reside on GPU/MPS pins the device in the .pt file
    # and makes cross-device loading slower (and in some MPS builds, fails).
    if isinstance(state_dict, dict):
        state_dict = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in state_dict.items()
        }

    torch.save(state_dict, path)

    # Always write the sidecar in lockstep with the .pt (even if empty),
    # using an atomic rename so readers never see a half-written file.
    sidecar_path = directory / f"{path.stem}.meta.json"
    tmp_sidecar = directory / f"{path.stem}.meta.json.tmp"
    try:
        tmp_sidecar.write_text(json.dumps(sidecar, indent=2, default=str),
                               encoding="utf-8")
        tmp_sidecar.replace(sidecar_path)  # atomic rename on POSIX
    except (TypeError, OSError) as exc:
        logger.warning("Failed to write sidecar metadata %s: %s", sidecar_path, exc)
        # Best-effort cleanup of a stray temp file so it doesn't linger
        try:
            if tmp_sidecar.exists():
                tmp_sidecar.unlink()
        except OSError:
            pass
    return path


def save_metadata(model_name: str, metadata: dict) -> Path:
    directory = model_dir(model_name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _load_checkpoint_payload(path: Path, device: torch.device) -> dict:
    """Load a checkpoint and return a dict with at least a ``state_dict`` key.

    Prefer the safe ``weights_only=True`` path. Old checkpoints were saved as
    ``{"state_dict": ..., "history": ..., ...}`` dicts which trigger an
    UnpicklingError under ``weights_only=True``; we detect any such failure,
    log a clear warning, and fall back to ``weights_only=False`` ONLY for
    those legacy files. New checkpoints written via save_checkpoint are
    pure state_dicts and load safely.

    Previously this used a fragile substring check ("weights_only" /
    "weightsunpickler") on the error message to decide whether to fall back.
    Future PyTorch releases may reword those strings. Instead: let real
    I/O errors propagate, but treat ANY other exception as "safe-mode
    rejection" and retry in legacy mode with a loud warning.
    """
    try:
        # New-format checkpoints: raw state_dict. weights_only=True is safe
        # because it refuses to deserialize arbitrary Python objects.
        raw = torch.load(path, map_location=device, weights_only=True)
    except (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError):
        # Real path/permission failure - must propagate unchanged.
        raise
    except OSError as exc:
        # Disk-level I/O failures (truncated file, EIO, read error) look
        # like safe-mode rejections to a naive except-Exception catch, which
        # would then retry under weights_only=False and either silently
        # "succeed" with corrupt weights or surface as a confusing
        # "legacy checkpoint" warning. Surface the true cause instead.
        logger.error("OS error reading checkpoint %s: %s", path.name, exc)
        raise
    except Exception as exc:
        logger.warning(
            "Safe-mode checkpoint load failed for %s (%s: %s); falling back "
            "to legacy pickled load. Re-save this checkpoint via save_checkpoint "
            "to clear this warning.",
            path.name, type(exc).__name__, exc,
        )
        raw = torch.load(path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "state_dict" in raw:
        # legacy wrapper
        return raw
    # new format: raw IS the state_dict
    return {"state_dict": raw}


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_name: str = "latest.pt",
    device: Optional[torch.device] = None,
) -> Tuple[object, dict]:
    device = device or get_device()
    path = checkpoint_path(model_name, checkpoint_name)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = _load_checkpoint_payload(path, device)
    model = create_model(model_name)
    # strict=False accommodates normalisation-layer refactors (e.g. CycleGAN
    # switching BatchNorm2d → InstanceNorm2d drops running_mean/running_var
    # keys). Missing or unexpected keys are logged so regressions stay visible.
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing:
        # Specifically call out BatchNorm running-stat keys - these show up
        # when an architecture saved with BatchNorm is loaded with
        # InstanceNorm (or vice-versa, via PatchDiscriminator norm toggles).
        # The resulting model loads "successfully" under strict=False but
        # silently runs with fresh BN stats, which can degrade output.
        bn_keys = [
            k for k in missing
            if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k
        ]
        if bn_keys:
            logger.warning(
                "Checkpoint %s is missing %d BatchNorm running-stat keys. This "
                "usually means the architecture was saved with InstanceNorm "
                "but loaded with BatchNorm (or vice-versa). Model output may "
                "be degraded. First 3: %s",
                path.name, len(bn_keys), bn_keys[:3],
            )
        logger.warning(
            "Checkpoint %s has %d missing keys (showing first 5): %s",
            path.name, len(missing), list(missing)[:5],
        )
    if unexpected:
        logger.warning(
            "Checkpoint %s has %d unexpected keys (showing first 5): %s",
            path.name, len(unexpected), list(unexpected)[:5],
        )
    model.to(device)
    model.eval()
    return model, payload
