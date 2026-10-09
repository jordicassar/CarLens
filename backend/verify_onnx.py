"""Check that the exported ONNX graph agrees with its PyTorch checkpoint.

    python verify_onnx.py [path/to/best.pt]

Run this after export_onnx.py. A max absolute difference around 1e-6 and an
identical top-5 mean the export is faithful; anything larger means the graph
does not match the model that produced the reported accuracy.
"""

import json
import sys

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import efficientnet_b0

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
ONNX_PATH = "app/carlens_b0.onnx"

with open("app/labels.json") as f:
    labels = json.load(f)

model = efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels))
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

x = torch.randn(1, 3, 224, 224)

with torch.inference_mode():
    torch_out = model(x).numpy()

session = ort.InferenceSession(ONNX_PATH)
onnx_out = session.run(["logits"], {"input": x.numpy()})[0]

print("classes:      ", len(labels))
print("output shape: ", onnx_out.shape)
print("max abs diff: ", np.abs(torch_out - onnx_out).max())
print("same top-5:   ", (torch_out.argsort()[0, -5:] == onnx_out.argsort()[0, -5:]).all())
