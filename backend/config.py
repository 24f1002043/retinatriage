import os
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


WEIGHTS_PATH = Path(
    os.environ.get("RETINATRIAGE_WEIGHTS", _BACKEND_DIR / "weights" / "model.pth")
)

FORCE_CPU = os.environ.get("RETINATRIAGE_FORCE_CPU", "").lower() in {"1", "true", "yes"}

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "RETINATRIAGE_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

MAX_UPLOAD_BYTES = _env_int("RETINATRIAGE_MAX_UPLOAD_MB", 10) * 1024 * 1024

ACCEPTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/tiff"}
