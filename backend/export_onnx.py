"""Export a fine-tuned EfficientNet-B0 checkpoint to ONNX.

    python export_onnx.py [path/to/best.pt]

The checkpoint is produced by the Colab notebook in notebooks/, which also
writes app/labels.json. Labels are read from that file rather than rebuilt
here, so the class order always matches the one the model was trained on.
Checkpoints are gitignored, so download best.pt from the notebook first.
"""

import json
import sys

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

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=18,
)

print(f"exported {ONNX_PATH} from {CHECKPOINT} with {len(labels)} classes")
print("weights may be written alongside as carlens_b0.onnx.data; ship both")
