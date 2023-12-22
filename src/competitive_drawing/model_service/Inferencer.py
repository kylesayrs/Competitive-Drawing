import threading
from PIL import Image

import torch

from competitive_drawing.model_service.opponent import (
    grid_search_stroke,
    SearchParameters,
    BezierCurve,
    get_uniform_ts
)
from competitive_drawing.model_service.utils.helpers import pil_to_input


def async_inference(method):
    def wrapper(self, *args, **kwargs):
        with self.mutex:
            ret = method(self, *args, **kwargs)

        return ret
        
    return wrapper


class Inferencer:
    """
    Wraps classifier model to handle classifier inference and opponent stroke
    inference. Uses a mutex to handle access to model resource

    Models are deployed in TensorRT rather than alternatives such as ORT because
    model gradients are necessary in order to optimize strokes for the AI opponent

    :param classifier_model: instance of classifier model
    """
    def __init__(self, classifier_model: torch.nn.Module):
        self._model = classifier_model
        self.mutex = threading.Semaphore(1)


    @async_inference
    def infer_image(self, image: Image):
        input = pil_to_input(image)
        with torch.no_grad():
            logits, _confidences = self._model(input)

        return logits[0].tolist()
    

    @async_inference
    def infer_stroke(
        self,
        image: Image,
        target_index: int,
        line_width: float,
        max_length: float,
    ):
        base_canvas = pil_to_input(image)[0][0]

        _loss, keypoints = grid_search_stroke(
            base_canvas,
            self._model,
            target_index,
            torch.optim.Adamax,
            { "lr": 0.03 },
            SearchParameters(
                max_width=line_width * 4,
                min_width=line_width
            ),
            max_length=max_length,
        )

        curve = BezierCurve(keypoints, num_approximations=20)
        stroke_samples = [
            curve.sample(t).cpu().detach().tolist()
            for t in get_uniform_ts(20)
        ]

        return stroke_samples


    @property
    def model(self):
        return self._model
