import re
import torch
import base64
from io import BytesIO
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor

from competitive_drawing import Settings
from competitive_drawing.train.classifier import Classifier

SETTINGS = Settings()


def get_classifier_model():
    """
    TODO: retrieve class definition in S3

    :return: classifier model class
    """
    return Classifier()


def imageDataUrlToImage(image_data_url: str) -> Image:
    """
    Convert a base64 image url to PIL Image

    :param image_data_url: base64 image url
    :return: image url as a PIL Image
    """
    image_data_str = re.sub("^data:image/.+;base64,", "", image_data_url)
    image_data = base64.b64decode(image_data_str)
    image_data_io = BytesIO(image_data)
    image = Image.open(image_data_io)

    return image


def pil_rgba_to_rgb(image: Image):
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])

    return background


def pil_to_input(image: Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = ImageOps.invert(image)
    first_channel = image.split()[0]

    input = to_tensor(first_channel)
    input = input.to(dtype=torch.float, device=SETTINGS.device)
    input = torch.unsqueeze(input, 0)  # singleton batch: (1, 1, image_size, image_size)

    return input
