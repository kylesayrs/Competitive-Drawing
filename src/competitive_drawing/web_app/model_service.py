from typing import Tuple, Union, List

import json
import requests

from competitive_drawing import Settings
from .utils import GAME_CONFIG



SETTINGS = Settings()


def server_infer(label_pair: Tuple[str, str], canvas_preview_data_url: str) -> List[float]:
    response = requests.post(
        f"{SETTINGS.model_service_base}/infer",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "gameConfig": GAME_CONFIG,
            "label_pair": label_pair,
            "imageDataUrl": canvas_preview_data_url
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

    return model_outputs
