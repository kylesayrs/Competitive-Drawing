from typing import Dict, Tuple, List

import torch

from competitive_drawing import Settings
from .Inferencer import Inferencer
from .utils import load_model, label_pair_to_str

SETTINGS = Settings()


class ModelManager():
    """
    Manages model inference such that . Handles the loading of model weights 

    1. if a game is running, the model should be loaded

    """
    def __init__(self):
        self.inferencers: Dict[str, torch.module.nn] = {}


    def start_model(self, label_pair: Tuple[str]):
        label_pair_str = label_pair_to_str(label_pair)

        if label_pair_str in self.inferencers:
            raise Exception("Model is already started")

        self.inferencers[label_pair_str] = Inferencer(load_model(label_pair_str))


    def get_inferencer(self, label_pair: Tuple[str, str]):
        label_pair_str = label_pair_to_str(label_pair)

        if label_pair_str not in self.inferencers:
            raise ValueError(
                f"Cannot get inferencer for {label_pair_str}. "
                f"Available inferencers are {self.inferencers.keys()}"
            )

        return self.inferencers[label_pair_str]


    def stop_model(self, label_pair: Tuple[str, str]):
        label_pair_str = label_pair_to_str(label_pair)
        del self.inferencers[label_pair_str]
