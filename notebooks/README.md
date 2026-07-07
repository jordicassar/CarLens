# Notebooks

Training and data prep for CarLens happen in **Google Colab**, not in this repo.
Colab gives us a free GPU and keeps the ~2GB Stanford Cars dataset off local disk.

## What lives in Colab (not here)

- **Data**: downloading the [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
  dataset via the Kaggle API. Your `kaggle.json` credential is **never** committed —
  it's uploaded into the Colab session at runtime (and is gitignored here as a safeguard).
- **Training**: transfer learning with a pretrained EfficientNet / ResNet backbone,
  fine-tuned on the Stanford Cars classes.
- **Export**: the trained weights (`.pt`/`.pth`) and the class-index → label map,
  pushed to the Hugging Face model hub. The FastAPI backend pulls them from there.

## Colab links

> Add the shareable Colab notebook URL(s) here once training starts.

- Training notebook: _TODO — paste Colab link_
- Data prep notebook: _TODO — paste Colab link (or fold into training)_

## Workflow

1. Open the Colab notebook, upload `kaggle.json`, pull the dataset.
2. Fine-tune the backbone; validate; export weights + `labels.json`.
3. Push weights to Hugging Face.
4. Point `backend/` at the HF weights and swap the stub for real inference (Phase 3).
