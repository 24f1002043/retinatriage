import logging
import time
from contextlib import asynccontextmanager
from io import BytesIO

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from config import ACCEPTED_IMAGE_TYPES, ALLOWED_ORIGINS, MAX_UPLOAD_BYTES
from gradcam import generate_cam
from model import CLASS_NAMES, DEVICE, get_model, preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("retinatriage")

_model = None
_untrained = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _untrained
    _model, _untrained = get_model()
    logger.info("RetinaTriage backend ready (device=%s)", DEVICE)
    yield
    logger.info("RetinaTriage backend shutting down")


app = FastAPI(
    title="RetinaTriage API",
    description=(
        "Diabetic retinopathy staging (5 classes) with Grad-CAM explainability. "
        "Research/demo use only, not a medical device."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "RetinaTriage API",
        "version": app.version,
        "docs": "/docs",
        "classes": CLASS_NAMES,
        "disclaimer": "Research/demo tool only, not for clinical or diagnostic use.",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "untrained": _untrained,
        "device": str(DEVICE),
    }


# returns grade, class probabilities and a Grad-CAM overlay
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Try again in a moment.",
        )

    if file.content_type and file.content_type.lower() not in ACCEPTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted types: {', '.join(sorted(ACCEPTED_IMAGE_TYPES))}."
            ),
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )

    try:
        pil_image = Image.open(BytesIO(contents))
        pil_image.verify()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the uploaded file as an image. "
                "Please upload a valid image (PNG, JPEG, etc.)."
            ),
        ) from exc

    started = time.perf_counter()
    try:
        input_tensor = preprocess(pil_image)

        with torch.no_grad():
            logits = _model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        pred_class = int(probs.argmax().item())
        confidence = round(float(probs[pred_class].item()), 4)
        prob_dict = {
            name: round(float(probs[i].item()), 4) for i, name in enumerate(CLASS_NAMES)
        }

        heatmap_b64 = generate_cam(_model, input_tensor, pred_class, pil_image)
    except Exception as exc:
        logger.exception("Prediction failed for upload %r", file.filename)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed while processing the image.",
        ) from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "Predicted stage %d (%s) conf=%.3f in %.1f ms",
        pred_class,
        CLASS_NAMES[pred_class],
        confidence,
        elapsed_ms,
    )

    return {
        "stage": pred_class,
        "stage_label": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "probabilities": prob_dict,
        "heatmap": heatmap_b64,
        "inference_ms": elapsed_ms,
        "untrained_warning": _untrained,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
