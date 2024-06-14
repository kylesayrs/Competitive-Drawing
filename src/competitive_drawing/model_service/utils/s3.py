import boto3
import torch
from io import BytesIO

from competitive_drawing import SETTINGS
from .helpers import get_classifier_model


def load_model(label_pair_str: str) -> torch.nn.Module:
    # get state dict from S3
    key = f"{SETTINGS.s3_models_root_folder}/{label_pair_str}/model.pth"
    state_dict_stream = _get_object_file_stream(SETTINGS.s3_models_bucket, key)
    state_dict = torch.load(state_dict_stream, map_location=SETTINGS.device)

    # instantiate model
    model = get_classifier_model()
    model.load_state_dict(state_dict)
    model = model.eval()

    return model


def _get_object_file_stream(bucket: str, key: str):
    obj = boto3.resource("s3").Object(bucket, key)
    tmp = obj.get()['Body'].read()
    return BytesIO(tmp)
