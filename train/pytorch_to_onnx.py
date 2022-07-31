import os
import sys
import torch

from train_model import Classifier

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    directory_path = os.path.dirname(checkpoint_path)

    model = Classifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    torch.onnx.export(
        model,
        torch.zeros((1, 1, 28, 28)),
        os.path.join(directory_path, "model.onnx"),
        input_names=["input"],
        output_names=["logits", "output"],
        opset_version=14
    )
