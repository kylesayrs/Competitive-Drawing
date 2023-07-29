import torch

from competitive_drawing.train.classifier import Classifier


def load_score_model(checkpoint_path: str):
    model = Classifier()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device("cpu"))
    )

    return model
