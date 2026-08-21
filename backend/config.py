"""
config.py — Environment-driven settings for the RetinaTriage API.

Every value has a sensible local-development default, so the app runs with
zero configuration, but can be pointed at different weights/origins in
deployment without touching code.
"""

import os
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back on anything unparseable."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
WEIGHTS_PATH = Path(
    os.environ.get("RETINATRIAGE_WEIGHTS", _BACKEND_DIR / "weights" / "model.pth")
)

# Force CPU inference even when a GPU is present (useful for small containers).
FORCE_CPU = os.environ.get("RETINATRIAGE_FORCE_CPU", "").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
# Comma-separated list, e.g. "http://localhost:5173,https://retinatriage.example"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "RETINATRIAGE_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

# Reject uploads larger than this before they reach the model (bytes).
MAX_UPLOAD_BYTES = _env_int("RETINATRIAGE_MAX_UPLOAD_MB", 10) * 1024 * 1024

ACCEPTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/tiff"}
