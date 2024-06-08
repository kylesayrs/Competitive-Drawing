from typing import Dict, Tuple

import copy
import torch

from competitive_drawing import Settings
from .Inferencer import Inferencer
from .utils import load_model, label_pair_to_str

SETTINGS = Settings()


class ModelManager():
    def __init__(self):
        self.inferencers: Dict[str, torch.module.nn] = {}  # maps label pairs to inferencers


    def scale(self, label_pair_games: Dict[str, int]):
        """
        Determines how to scale inferences with regards to the number of active
        games for each label pair. The current policy is to have exactly one
        inferencer for each active label pair, no matter how many games. Future
        policies may scale linearly with the number of games or mutex delay time.

        :param label_pair_games: Dictionary mapping label pair strings to number
            of active games
        """
        # scale up
        for label_pair_str, num_games in label_pair_games.items():
            if num_games > 0 and label_pair_str not in self.inferencers:
                self.start_inferencer(label_pair_str)

        # scale down
        inferencer_items = list(self.inferencers.items())  # cache to avoid iterable resizing
        for label_pair_str, inferencer in inferencer_items:
            if label_pair_str not in label_pair_games or label_pair_games[label_pair_str] <= 0:
                self.stop_inferencer(label_pair_str)


    def start_inferencer(self, label_pair_str: str):
        self.inferencers[label_pair_str] = Inferencer(load_model(label_pair_str))


    def stop_inferencer(self, label_pair_str: str):
        del self.inferencers[label_pair_str]


    def get_inferencer(self, label_pair: Tuple[str, str]):
        label_pair_str = label_pair_to_str(label_pair)

        if label_pair_str not in self.inferencers:
            raise ValueError(
                f"Cannot get inferencer for {label_pair_str}. "
                f"Available inferencers are {self.inferencers.keys()}"
            )

        return self.inferencers[label_pair_str]
    
