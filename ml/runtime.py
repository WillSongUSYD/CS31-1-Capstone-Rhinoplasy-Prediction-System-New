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
    directory = model_dir(model_name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    torch.save(payload, path)
    return path


def save_metadata(model_name: str, metadata: dict) -> Path:
    directory = model_dir(model_name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_name: str = "latest.pt",
    device: Optional[torch.device] = None,
) -> Tuple[object, dict]:
    device = device or get_device()
    path = checkpoint_path(model_name, checkpoint_name)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False is required here because our checkpoints contain
    # training history dicts alongside the state_dict. Checkpoints are only
    # loaded from the local `models/` directory that we produced ourselves,
    # so arbitrary-code deserialization risk is bounded to our own files.
    payload = torch.load(path, map_location=device, weights_only=False)
    model = create_model(model_name)
    # strict=False accommodates normalisation-layer refactors (e.g. CycleGAN
    # switching BatchNorm2d → InstanceNorm2d drops running_mean/running_var
    # keys). Missing or unexpected keys are logged so regressions stay visible.
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing:
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
