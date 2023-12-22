from typing import Tuple, Dict, List

import json
import requests

from competitive_drawing import Settings
from .utils import GAME_CONFIG

SETTINGS = Settings()
HEADERS = {"Content-Type": "application/json"}


def server_infer(label_pair: Tuple[str, str], preview_image_data_url: str) -> Tuple[float, float]:
    """
    Requence inference from a model service server

    :param label_pair: labels which define the model/game
    :param preview_image_data_url: data url of preview image
    :return: model outputs
    """
    response = requests.post(
        f"{SETTINGS.model_service_base}/infer",
        headers=HEADERS,
        data=json.dumps({
            "gameConfig": GAME_CONFIG,
            "label_pair": label_pair,
            "imageDataUrl": preview_image_data_url
        })
    )

    if (not response.ok):
        raise ValueError(f"Invalid response {response}")
    
    response_json = response.json()
    if ("modelOutputs" not in response_json):
        raise ValueError(f"Invalid response body {response.content()}")
    
    model_outputs = response_json["modelOutputs"]
    if (model_outputs is None or len(model_outputs) != 2):
        raise ValueError(f"Invalid response body {response.content()}")

    return tuple(model_outputs)


def server_update(num_games_by_label_pair_str: Dict[str, int]):
    response = requests.post(
        f"{SETTINGS.model_service_base}/games",
        headers=HEADERS,
        data=json.dumps({
            "label_pair_games": num_games_by_label_pair_str
        }),
    )

    if (not response.ok):
        raise ValueError(f"Invalid response {response}")
