import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

def _jpeg_bytes(size=(256, 256), color=(120, 130, 140)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_returns_five_scored_labels():
    r = client.post("/predict", files={"file": ("car.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200

    preds = r.json()["predictions"]
    assert len(preds) == 5
    assert all(0.0 <= p["confidence"] <= 1.0 for p in preds)
    assert sum(p["confidence"] for p in preds) <= 1.0 + 1e-6
    assert preds == sorted(preds, key=lambda p: p["confidence"], reverse=True)

def test_rejects_non_image_content_type():
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415

def test_rejects_undecodable_image():
    r = client.post("/predict", files={"file": ("fake.jpg", b"not a jpeg", "image/jpeg")})
    assert r.status_code == 400

def test_rejects_oversized_upload():
    big = b"\xff" * (8 * 1024 * 1024 + 1)
    r = client.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


