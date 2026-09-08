import numpy as np
import onnxruntime as ort
import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
model.eval()

x = torch.randn(1, 3, 224, 224)

with torch.inference_mode():
    torch_out = model(x).numpy()

session = ort.InferenceSession("app/efficientnet_b0.onnx")
onnx_out = session.run(["logits"], {"input": x.numpy()})[0]

print("max abs diff:", np.abs(torch_out - onnx_out).max())
print("same top-5:", (torch_out.argsort()[0, -5:] == onnx_out.argsort()[0, -5:]).all())