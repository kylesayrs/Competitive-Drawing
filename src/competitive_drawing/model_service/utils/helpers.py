import re
import torch
import pickle
import base64
from io import BytesIO
from PIL import Image

from .s3 import get_object_file_stream
from competitive_drawing import Settings
from competitive_drawing.train.utils.model import Classifier


def get_model_class():
    """
    I can't figure out how to get pickling the original class to work
    so this is my lazy solution
    """
    return Classifier


    bucket = Settings.get("S3_MODELS_BUCKET", "competitive-drawing-models-prod")

    root_folder = Settings.get("S3_MODELS_ROOT_FOLDER", "static_crop_50x50")
    key = f"{root_folder}/model.pkl"

    pickled_file_stream = get_object_file_stream(bucket, key)
    return pickle.loads(torch.load(pickled_file_stream))


def imageDataUrlToImage(imageDataUrl):
    image_data_url = imageDataUrl
    image_data_str = re.sub("^data:image/.+;base64,", "", image_data_url)
    image_data = base64.b64decode(image_data_str)
    image_data_io = BytesIO(image_data)
    image = Image.open(image_data_io)

    return image
