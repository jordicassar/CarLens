import logging
import time
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from app.predictor import predict

logger = logging.getLogger("carlens")

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="CarLens", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(file: Annotated[UploadFile, File()]):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type {file.content_type!r}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 8 MB limit")

    started = time.perf_counter()
    try:
        predictions = predict(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image")
    elapsed_ms = (time.perf_counter() - started) * 1000

    logger.info("predict ok bytes=%d ms=%.1f", len(image_bytes), elapsed_ms)
    return {"predictions": predictions, "latency_ms": round(elapsed_ms, 1)}

