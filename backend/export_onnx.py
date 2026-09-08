import json

import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
model.eval()

# Print the exact preprocessing you'll have to reproduce by hand
print(weights.transforms())

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "app/efficientnet_b0.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)

with open("app/labels.json", "w") as f:
    json.dump(weights.meta["categories"], f)

print("exported", len(weights.meta["categories"]), "labels")
