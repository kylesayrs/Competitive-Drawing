from typing import Tuple

import re
import base64
from PIL import Image
from io import BytesIO

from competitive_drawing import Settings

SETTINGS = Settings()


def label_pair_to_str(label_pair: Tuple[str, str]) -> str:
    label_pair = label_pair.copy()
    label_pair.sort()
    return "-".join(label_pair)



def data_url_to_image(data_url):
    image_data_str = re.sub("^data:image/.+;base64,", "", data_url)
    image_data = base64.b64decode(image_data_str)
    image_data_io = BytesIO(image_data)
    return Image.open(image_data_io)


GAME_CONFIG = {
    "softmaxFactor": SETTINGS.softmax_factor,
    "canvasSize": SETTINGS.canvas_size,
    "canvasLineWidth": SETTINGS.canvas_line_width,
    "imageSize": SETTINGS.image_size,
    "imagePadding": SETTINGS.image_padding,
    "distancePerTurn": SETTINGS.distance_per_turn,
    "staticCrop": SETTINGS.static_crop
}
