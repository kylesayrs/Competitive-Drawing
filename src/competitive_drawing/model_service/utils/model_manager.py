from typing import Dict, Tuple, List

import torch
from .helpers import get_model_class
from .s3 import get_object_file_stream
from .inference import Inferencer
from competitive_drawing import Settings

SETTINGS = Settings()


class ModelManager():
    def __init__(self):
        self.model_class = get_model_class()
        self.inferencers: Dict[Tuple[str, str], torch.module.nn] = {}


    def start_model(self, label_pair: Tuple[str]):
        label_pair_str = self._label_pair_to_str(label_pair)
        if label_pair_str in self.inferencers:
            raise Exception("Model is already started")

        key = f"{SETTINGS.s3_models_root_folder}/{label_pair_str}/model.pth"
        state_dict_stream = get_object_file_stream(SETTINGS.s3_models_bucket, key)

        self.inferencers[label_pair_str] = Inferencer(
            self.model_class,
            torch.load(state_dict_stream, map_location=SETTINGS.device),
        )


    def get_inferencer(self, label_pair: Tuple[str, str]):
        label_pair_str = self._label_pair_to_str(label_pair)

        if label_pair_str not in self.inferencers:
            raise ValueError(
                f"Cannot get inferencer for {label_pair_str}. "
                f"Available inferencers are {self.inferencers.keys()}"
            )

        return self.inferencers[label_pair_str]


    def stop_model(self, label_pair: Tuple[str, str]):
        label_pair_str = self._label_pair_to_str(label_pair)
        del self.inferencers[label_pair_str]


    def _label_pair_to_str(self, label_pair: Tuple[str, str]):
        label_pair = label_pair.copy()
        label_pair.sort()
        return "-".join(label_pair)
