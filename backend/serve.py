import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ml.config import BENCHMARK_PATH, PREDICTIONS_DIR, REPO_ROOT

from .db import fetch_history, insert_history
from .inference import run_prediction, save_upload
from .schemas import PredictResponse

logger = logging.getLogger(__name__)

DIST_DIR = REPO_ROOT / "frontend" / "dist"
DIST_ASSETS_DIR = DIST_DIR / "assets"

# Whitelist of valid model names (prevents path injection via model_name)
VALID_MODEL_NAMES = {
    "autoencoder", "pix2pix", "cyclegan", "diffusion",
    "autoencoder_nose", "pix2pix_nose", "cyclegan_nose", "diffusion_nose",
}


def _parse_origins(raw: str) -> List[str]:
    """Parse a comma-separated CORS origin list, rejecting invalid entries.

    Rules:
      - Reject wildcard `*` outright (combining wildcard with allow_credentials
        is a browser-level error and is dangerous even without credentials).
      - Reject blank / whitespace-only entries.
      - Require http/https scheme + netloc, no path/query/fragment.
    Rejected entries are logged at WARNING level and skipped; we never raise,
    so a stray malformed entry won't prevent the server from starting.
    """
    valid: List[str] = []
    for entry in (raw or "").split(","):
        origin = entry.strip()
        if not origin:
            continue
        if origin == "*":
            logger.warning("Rejecting wildcard '*' CORS origin (unsafe with credentials)")
            continue
        try:
            parsed = urlparse(origin)
        except ValueError:
            logger.warning("Rejecting malformed CORS origin %r", origin)
            continue
        if parsed.scheme not in ("http", "https"):
            logger.warning("Rejecting CORS origin %r: scheme must be http/https", origin)
            continue
        if not parsed.netloc:
            logger.warning("Rejecting CORS origin %r: missing host", origin)
            continue
        # An origin is scheme://host[:port] only - a path/query/fragment is
        # invalid per the CORS spec.
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            logger.warning("Rejecting CORS origin %r: must not contain path/query/fragment", origin)
            continue
        # Canonicalise: drop trailing slash if present.
        canonical = f"{parsed.scheme}://{parsed.netloc}"
        valid.append(canonical)
    return valid


# CORS: read allowed origins from env var, default to common dev origins
_cors_env = os.getenv("CS31_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
CORS_ORIGINS = _parse_origins(_cors_env)

app = FastAPI(title="CS31 Rhinoplasty Outcome Prediction")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Only expose prediction outputs, not the entire artifacts tree (which contains
# model checkpoints, training benchmarks, and raw prepared pairs that should
# not be publicly downloadable).
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/predictions", StaticFiles(directory=str(PREDICTIONS_DIR)), name="predictions")
if DIST_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_ASSETS_DIR)), name="frontend-assets")


def _artifact_url(path: Optional[Path]) -> Optional[str]:
    """Convert a filesystem path to a URL served under /predictions/.

    Returns None for any path outside PREDICTIONS_DIR - this is expected for
    historical records written before the mount point was tightened, so we
    use debug-level logging to avoid flooding logs on /api/history calls.
    """
    if path is None:
        return None
    try:
        rel = Path(path).resolve().relative_to(PREDICTIONS_DIR.resolve())
    except (ValueError, OSError):
        logger.debug("Path outside PREDICTIONS_DIR, skipping URL: %s", path)
        return None
    return "/predictions/" + str(rel).replace("\\", "/")


def _validate_model_name(model_name: str) -> str:
    if model_name not in VALID_MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Allowed: {sorted(VALID_MODEL_NAMES)}",
        )
    return model_name


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "cs31-backend"}


@app.get("/api/history")
def history() -> list[dict]:
    rows = list(fetch_history())
    for row in rows:
        row["input_url"] = _artifact_url(Path(row["input_path"])) if row.get("input_path") else None
        row["pre_url"] = _artifact_url(Path(row["pre_path"])) if row.get("pre_path") else None
        row["generated_url"] = _artifact_url(Path(row["generated_post_path"])) if row.get("generated_post_path") else None
        row["reference_url"] = (
            _artifact_url(Path(row["reference_post_path"]))
            if row.get("reference_post_path")
            else None
        )
    return rows


@app.get("/api/benchmarks")
def benchmarks() -> list[dict]:
    if not BENCHMARK_PATH.exists():
        return []
    frame = pd.read_csv(BENCHMARK_PATH)
    return frame.to_dict("records")


@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("pix2pix"),
    paired_input: bool = Form(True),
):
    _validate_model_name(model_name)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")
    try:
        # save_upload does PIL decoding + disk I/O; off-load so we don't
        # block the event loop for large uploads.
        upload_path = await run_in_threadpool(
            save_upload, file_bytes, file.filename or "input.png"
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        # run_prediction does GPU inference + image encoding, potentially
        # hundreds of ms to seconds. Off-load so concurrent requests aren't
        # blocked.
        result = await run_in_threadpool(
            run_prediction,
            upload_path=upload_path,
            model_name=model_name,
            paired_input=paired_input,
        )
    except FileNotFoundError as exc:
        logger.warning("Checkpoint not found: %s", exc)
        raise HTTPException(status_code=404, detail="Model checkpoint not found.") from exc
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal error during prediction.")

    # insert_history blocks on SQLite I/O; off-load.
    await run_in_threadpool(
        insert_history,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "input_mode": result["input_mode"],
            "input_path": str(upload_path),
            "pre_path": str(result["pre_path"]),
            "reference_post_path": str(result["reference_post_path"]) if result["reference_post_path"] else None,
            "generated_post_path": str(result["generated_post_path"]),
            "status": "completed",
            "notes": "",
        },
    )

    return PredictResponse(
        model_name=model_name,
        input_mode=result["input_mode"],
        uploaded_input_url=_artifact_url(upload_path),
        pre_image_url=_artifact_url(result["pre_path"]),
        generated_post_url=_artifact_url(result["generated_post_path"]),
        reference_post_url=_artifact_url(result["reference_post_path"]),
        metrics=result["metrics"],
        description=result.get("description"),
        landmarks=result.get("landmarks"),
        disclaimer="Research prototype only. Not for medical decision-making.",
    )


@app.get("/api/training-history/{model_name}")
def training_history(model_name: str) -> dict:
    _validate_model_name(model_name)
    history_path = REPO_ROOT / "models" / "outcome" / model_name / "history.json"
    if not history_path.exists():
        return {"model": model_name, "history": []}
    try:
        return {"model": model_name, "history": json.loads(history_path.read_text())}
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read training history for %s", model_name)
        return {"model": model_name, "history": []}


@app.get("/")
def root():
    dist_index = DIST_DIR / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index)
    return JSONResponse(
        {
            "message": "Frontend build not found. Run the backend API at /api/* or build the React frontend in frontend/.",
        }
    )


def main() -> None:
    import uvicorn
    uvicorn.run("backend.serve:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
