# CarLens

![CI](https://github.com/jordicassar/CarLens/actions/workflows/ci.yml/badge.svg)

A full-stack image classification app. A FastAPI service runs an EfficientNet-B0
image classifier and returns the top-5 predictions for an uploaded photo, with a
React Native (Expo) client planned on top of it.

The model currently served is the stock **ImageNet-1k** EfficientNet-B0, so it
returns generic categories such as `sports car` or `pickup` rather than specific
makes and models. Fine-tuning on
[Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) to get
make/model/year predictions is the next phase. The serving pipeline is built and
tested first so that swapping in the fine-tuned weights is a single-file change.

## Status

| Phase | State |
| --- | --- |
| 1. Scaffold: monorepo, CI, tests | Done |
| 2. Train: fine-tune on Stanford Cars in Colab | Not started |
| 3. Inference: real `/predict` endpoint | Done (ImageNet baseline) |
| 4. Frontend: image picker and results screen | Not started |
| 5. Deploy: public hosted endpoint | Not started |

## How inference works

Training and serving are deliberately separated, and they do not share a runtime.

**PyTorch is used only at export time.** `backend/export_onnx.py` loads
EfficientNet-B0 and writes out an ONNX graph plus the ImageNet label map.
`backend/verify_onnx.py` then checks that the exported graph agrees with the
original PyTorch model (max absolute difference is on the order of `1e-6`).

**ONNX Runtime is what actually serves requests.** `backend/app/predictor.py`
imports no PyTorch at all. It loads the `.onnx` graph, applies the preprocessing
by hand in NumPy (resize shortest side to 256 bicubic, center crop 224,
normalize with the ImageNet mean and standard deviation), runs the session, and
applies softmax.

The reason for the split: PyTorch is a training framework carrying autograd,
optimizers, and CUDA kernels, none of which are used at inference. Dropping it
from the runtime removes roughly a gigabyte of dependencies and makes the
service small enough to host on a free tier.

## Architecture

```
CarLens/
├── backend/
│   ├── app/
│   │   ├── main.py                    FastAPI routes and upload validation
│   │   ├── predictor.py               ONNX Runtime inference, NumPy preprocessing
│   │   ├── efficientnet_b0.onnx       exported graph
│   │   ├── efficientnet_b0.onnx.data  exported weights
│   │   └── labels.json                1000 ImageNet class names
│   ├── tests/test_api.py              API contract and rejection paths
│   ├── export_onnx.py                 produces the .onnx artifact
│   ├── verify_onnx.py                 checks it against PyTorch
│   ├── Dockerfile                     runs as UID 1000, listens on 7860
│   ├── requirements.txt               runtime only, no PyTorch
│   └── requirements-dev.txt           tests, lint, and export tooling
├── frontend/                          Expo app, still the starter template
├── notebooks/                         pointer to the Colab training work
└── .github/workflows/ci.yml           ruff and pytest on push and PR
```

## Running the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for interactive API docs.

- `GET /health` returns a liveness check.
- `POST /predict` takes a multipart image upload and returns the top-5 labels
  with confidences and the inference latency in milliseconds.

Uploads are validated on content type, size (8 MB cap), and decodability, and
rejected with 415, 413, and 400 respectively.

Example response:

```json
{
  "predictions": [
    {"label": "sports car", "confidence": 0.7708},
    {"label": "car wheel", "confidence": 0.0343},
    {"label": "racer", "confidence": 0.0322},
    {"label": "grille", "confidence": 0.0176},
    {"label": "convertible", "confidence": 0.0097}
  ],
  "latency_ms": 44.7
}
```

## Tests and lint

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

Five tests cover the API contract: health, a well-formed prediction (five
results, confidences in range, sorted, summing to at most 1.0), and the three
rejection paths. CI runs both on every push and pull request.

## Docker

```bash
cd backend
docker build -t carlens .
docker run -p 7860:7860 carlens
```

The container runs as UID 1000 and listens on port 7860. Model weights are baked
into the image at build time so that cold starts do not stall on a download.

## Regenerating the model artifact

The `.onnx` files are committed, so this is only needed when changing the model.

```bash
cd backend
pip install -r requirements-dev.txt
python export_onnx.py
python verify_onnx.py
```

`verify_onnx.py` should report a maximum absolute difference near `1e-6` and an
identical top-5 against PyTorch.

## Frontend

```bash
cd frontend
npm install
npx expo start
```

Currently the unmodified Expo starter template. Wiring it to `/predict` is
phase 4.

## Notebooks

No training code lives in this repo. See [notebooks/README.md](notebooks/README.md)
for the Colab workflow and links.
