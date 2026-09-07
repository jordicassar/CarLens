import json
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

_HERE = Path(__file__).parent

_session = ort.InferenceSession(str(_HERE / "efficientnet_b0.onnx"))
_labels = json.loads((_HERE / "labels.json").read_text())

RESIZE_SIZE = 256
CROP_SIZE = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TOP_K = 5


def _preprocess(image: Image.Image) -> np.ndarray:
    # Shortest side to 256, aspect ratio preserved, bicubic
    w, h = image.size
    scale = RESIZE_SIZE / min(w, h)
    image = image.resize(
        (round(w * scale), round(h * scale)), Image.Resampling.BICUBIC
    )

    # Center-crop 224x224
    w, h = image.size
    left = int(round((w - CROP_SIZE) / 2.0))
    top = int(round((h - CROP_SIZE) / 2.0))
    image = image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    # HWC uint8 -> normalized CHW float32, plus batch dim
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...].astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(image_bytes: bytes, top_k: int = TOP_K):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    batch = _preprocess(image)

    logits = _session.run(["logits"], {"input": batch})[0][0]
    probs = _softmax(logits)

    indices = probs.argsort()[-top_k:][::-1]
    return [
        {"label": _labels[i], "confidence": round(float(probs[i]), 4)}
        for i in indices
    ]
