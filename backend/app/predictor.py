from io import BytesIO

import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

_weights = EfficientNet_B0_Weights.IMAGENET1K_V1
_model = efficientnet_b0(weights=_weights)
_model.eval()

_preprocess = _weights.transforms()
_labels = _weights.meta["categories"]

TOP_K = 5


@torch.inference_mode()
def predict(image_bytes: bytes, top_k: int = TOP_K):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    batch = _preprocess(image).unsqueeze(0)

    logits = _model(batch)
    probs = logits.softmax(dim=1).squeeze(0)

    scores, indices = probs.topk(top_k)
    return [
        {"label": _labels[i], "confidence": round(s.item(), 4)}
        for s, i in zip(scores, indices)
    ]