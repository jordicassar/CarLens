# CarLens
A full-stack ML app that classifies car make and model from a photo. EfficientNet/ResNet transfer learning (PyTorch, trained on Stanford Cars in Colab) served through a FastAPI inference endpoint on Hugging Face Spaces, with a React Native (Expo) mobile front end.

## Architecture

```
CarLens/
├── backend/      FastAPI inference service (stubbed /predict)
├── frontend/     React Native (Expo) app — one screen for now
├── notebooks/    Pointer to the Colab training work (no training code here)
├── models/       Trained weights land here at runtime — gitignored, live on HF
└── .gitignore    Python, Node, model weights, and credentials (kaggle.json etc.)
```

- **ML**: PyTorch transfer learning (EfficientNet / ResNet) on the
  [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset.
  Training runs in **Google Colab** (GPU), not locally — see [notebooks/](notebooks/).
- **Backend**: FastAPI, deployed to **Hugging Face Spaces**. Weights are pulled
  from the HF model hub, not stored in git.
- **Frontend**: Expo (React Native), talks to the backend `/predict` endpoint.

## 5-Phase Plan

1. **Scaffold** (this repo) — monorepo structure, stubbed endpoints, placeholder UI.
2. **Train** — transfer-learn EfficientNet/ResNet on Stanford Cars in Colab;
   export weights + label map; push weights to Hugging Face.
3. **Real inference** — wire the FastAPI `/predict` endpoint to the trained model
   (load weights from HF, preprocess image, return real top-5).
4. **Frontend integration** — image picker → upload to `/predict` → render results.
5. **Deploy** — ship the backend to Hugging Face Spaces; point the app at it.

## Running each piece

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Then open http://127.0.0.1:8000/docs for the interactive API docs.
- `GET /health` → liveness check
- `POST /predict` → returns **fake** top-5 predictions (multipart image upload)

### Frontend
```bash
cd frontend
npm install
npx expo start
```
Scan the QR code with Expo Go, or press `i` / `a` for a simulator.

### Notebooks
No training code lives here — see [notebooks/README.md](notebooks/README.md) for
the Colab links and workflow.
