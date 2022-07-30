from train_model import Classifier

import torch

model = Classifier()
model.load_state_dict(torch.load("./checkpoints/1oqkgf1z/epoch9.pth"))

torch.onnx.export(
    model,
    torch.zeros((1, 1, 28, 28)),
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=14
)
