from typing import Tuple

import os
import torch
from dotenv import load_dotenv
from .helpers import get_model_class
from .s3 import get_object_file_stream
from .inference import Inferencer

load_dotenv(".env")
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class ModelManager():
    def __init__(self):
        self.model_class = get_model_class()
        self.models = {}  # maps label pairs to models


    def start_model(self, label_pair):
        label_pair_str = self._label_pair_to_str(label_pair)
        if label_pair_str in self.models:
            raise Exception("Model is already started")

        bucket = os.environ.get("S3_MODELS_BUCKET", "competitive-drawing-models-prod")
        root_folder = os.environ.get("S3_ROOT_FOLDER", "static_crop_50x50")
        key = f"{root_folder}/{label_pair_str}/model.pth"
        state_dict_stream = get_object_file_stream(bucket, key)

        self.models[label_pair_str] = Inferencer(self.model_class, torch.load(state_dict_stream))


    def get_model(self, label_pair):
        label_pair_str = self._label_pair_to_str(label_pair)
        return self.models[label_pair_str]


    def stop_model(self, label_pair):
        label_pair_str = self._label_pair_to_str(label_pair)
        del self.models[label_pair_str]


    def _label_pair_to_str(self, label_pair: Tuple[str, str]):
        label_pair = label_pair.copy()
        label_pair.sort()
        return "-".join(label_pair)
