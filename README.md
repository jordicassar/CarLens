# CarLens

![CI](https://github.com/jordicassar/CarLens/actions/workflows/ci.yml/badge.svg)

**[Try the live demo](https://carlens-qmnn.onrender.com/)** &nbsp;·&nbsp;
[API docs](https://carlens-qmnn.onrender.com/docs)

> Hosted on Render's free tier, which spins the container down after about
> 15 minutes of inactivity. The first request after an idle period takes
> 30 to 60 seconds to wake it. Subsequent requests respond in under a second.

A full-stack image classification app. A FastAPI service identifies the make,
model and year of a car from a photo and returns the top-5 predictions, with a
web page and a React Native (Expo) client on top of it.

The model is an EfficientNet-B0 fine-tuned on
[Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), which
covers **196 classes** at the granularity of `Ferrari 458 Italia Coupe 2012`.
It reaches **0.671 top-1 and 0.886 top-5** on the held-out test split. Training
runs in Google Colab; serving runs on ONNX Runtime with no PyTorch in the
container.

Every class in the dataset is a 2012 or earlier model, and the model always
returns a prediction, so a newer or unusual vehicle will still produce a
confident-looking answer. See [Evaluation](#evaluation) for what it gets wrong
and why.

## Status

| Phase | State |
| --- | --- |
| 1. Scaffold: monorepo, CI, tests | Done |
| 2. Train: fine-tune on Stanford Cars in Colab | Done (test top-1 0.671, top-5 0.886) |
| 3. Inference: real `/predict` endpoint | Done |
| 4. Frontend: image picker and results screen | Done (web page and Expo app) |
| 5. Deploy: public hosted endpoint | Done (Render, free tier) |

## How inference works

Training and serving are deliberately separated, and they do not share a runtime.

**PyTorch is used only for training and export.** The Colab notebook in
[notebooks/](notebooks/) fine-tunes EfficientNet-B0 and exports an ONNX graph
plus the 196-class label map, after checking that the exported graph agrees
with the PyTorch model it came from (max absolute difference on the order of
`1e-6`).

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
│   │   ├── static/index.html          landing page served at /
│   │   ├── carlens_b0.onnx            exported graph
│   │   ├── carlens_b0.onnx.data       exported weights
│   │   └── labels.json                196 Stanford Cars class names
│   ├── tests/test_api.py              API contract and rejection paths
│   ├── export_onnx.py                 produces the .onnx artifact
│   ├── verify_onnx.py                 checks it against PyTorch
│   ├── Dockerfile                     runs as UID 1000, listens on 7860
│   ├── requirements.txt               runtime only, no PyTorch
│   └── requirements-dev.txt           tests, lint, and export tooling
├── frontend/App.js                    Expo app: picker, upload, results
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
    {"label": "Ferrari 458 Italia Coupe 2012", "confidence": 0.8142},
    {"label": "Ferrari 458 Italia Convertible 2012", "confidence": 0.0913},
    {"label": "Ferrari California Convertible 2012", "confidence": 0.0241},
    {"label": "Lamborghini Gallardo LP 570-4 Superleggera 2012", "confidence": 0.0118},
    {"label": "McLaren MP4-12C Coupe 2012", "confidence": 0.0076}
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

There are two clients, both talking to the same `/predict` endpoint.

**Web.** `backend/app/static/index.html` is served at `/` by FastAPI. It is a
single self-contained page with no build step and no dependencies: drop in a
photo, see the top-5 with confidence bars.

**Mobile.** A React Native app built with Expo (SDK 57).

```bash
cd frontend
npm install
npx expo start
```

Scan the QR code with Expo Go. On networks that isolate clients, such as campus
Wi-Fi, use `npx expo start --tunnel` instead.

The app picks a photo from the library or the camera, uploads it, and renders
the same top-5 view as the web page. Two details worth noting:

- It pings `/health` on launch, because the free tier sleeps when idle. The
  server is usually awake by the time a photo has been chosen. A failed upload
  is retried once, and the spinner explains the wait after four seconds.
- Uploads are sent as an `expo-file-system` `File`. Expo SDK 57 replaces the
  global `fetch` with a standards-compliant implementation that accepts only
  real Blobs in `FormData`, so the older React Native `{ uri, name, type }`
  object fails with "Unsupported FormDataPart implementation".

## Notebooks

No training code lives in this repo. See [notebooks/README.md](notebooks/README.md)
for the Colab workflow and links.
