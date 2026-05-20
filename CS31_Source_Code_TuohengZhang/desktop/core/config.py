"""Apply environment-variable overrides so that the existing
``backend.inference`` / ``ml.*`` modules pick up bundle-relative paths
when they initialise.

``install_environment()`` MUST be called BEFORE any ``import backend.*``
or ``import ml.*`` — the target modules read these env vars at import
time (``_sd_artefact_paths`` in backend/inference.py uses ``os.environ``
eagerly). desktop.app.main() handles ordering.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from .paths import (
    bundle_insightface,
    bundle_lora,
    user_config_path,
    user_predictions_dir,
    user_sd_base_dir,
)

logger = logging.getLogger(__name__)


def install_environment() -> None:
    """Set env vars BEFORE project-code import.

    Each assignment is guarded with ``setdefault`` so that a developer
    running ``python -m desktop`` with a shell override (e.g. pointing
    ``CS31_SD_BASE_DIR`` at a local dev copy) isn't silently overwritten.
    """
    # Where backend.inference._sd_artefact_paths() reads SD base/LoRA from.
    os.environ.setdefault("CS31_SD_BASE_DIR", str(user_sd_base_dir()))
    os.environ.setdefault("CS31_SD_LORA_DIR", str(bundle_lora()))

    # Where ml.config.PREDICTIONS_DIR is rerouted to (needs the 2-line
    # patch in ml/config.py - see plan). Harmless if the project hasn't
    # applied that patch yet; nothing tries to write until inference runs.
    os.environ.setdefault("CS31_PREDICTIONS_DIR", str(user_predictions_dir()))

    # InsightFace looks for its ONNX models here before trying to download.
    # We ship buffalo_l inside the bundle, so this MUST point at the
    # bundled copy for offline first-launch to work.
    os.environ.setdefault("INSIGHTFACE_HOME", str(bundle_insightface()))

    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

    logger.info(
        "env: SD_BASE=%s LORA=%s PREDICTIONS=%s INSIGHTFACE=%s HF=%s",
        os.environ["CS31_SD_BASE_DIR"], os.environ["CS31_SD_LORA_DIR"],
        os.environ["CS31_PREDICTIONS_DIR"], os.environ["INSIGHTFACE_HOME"],
        os.environ["HF_ENDPOINT"],
    )


# ---------------------------------------------------------------------------
# Persistent user config (force_cpu decision, etc.)
# ---------------------------------------------------------------------------


@dataclass
class UserConfig:
    """Small JSON-persisted config. Grows over time; keep the schema
    versionless-but-additive (old fields keep defaulting to safe values)."""
    force_cpu: bool = False
    seen_onboarding: bool = False


def load_user_config() -> UserConfig:
    p = user_config_path()
    if not p.exists():
        return UserConfig()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Corrupt config at %s (%s); resetting", p, exc)
        return UserConfig()
    return UserConfig(
        force_cpu=bool(raw.get("force_cpu", False)),
        seen_onboarding=bool(raw.get("seen_onboarding", False)),
    )


def save_user_config(cfg: UserConfig) -> None:
    p = user_config_path()
    data = {"force_cpu": cfg.force_cpu, "seen_onboarding": cfg.seen_onboarding}
    # Atomic write so a crash mid-save doesn't corrupt the file.
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)
