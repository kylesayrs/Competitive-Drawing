import re
import torch
import pickle
import base64
from io import BytesIO
from PIL import Image

from .s3 import get_object_file_stream
from competitive_drawing import Settings
from competitive_drawing.train.classifier import Classifier


def get_model_class():
    return Classifier


def imageDataUrlToImage(image_data_url):
    image_data_str = re.sub("^data:image/.+;base64,", "", image_data_url)
    image_data = base64.b64decode(image_data_str)
    image_data_io = BytesIO(image_data)
    image = Image.open(image_data_io)

    return image
