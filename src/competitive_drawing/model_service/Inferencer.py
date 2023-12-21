import numpy
import threading
from PIL import Image

import torch
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from competitive_drawing import Settings
from competitive_drawing.model_service.opponent import (
    grid_search_stroke,
    SearchParameters,
    BezierCurve,
    get_uniform_ts
)
from competitive_drawing.model_service.utils.helpers import pil_to_input
from .utils.helpers import pil_rgba_to_rgb


def async_inference(method):
    def wrapper(self, *args, **kwargs):
        with self.semaphore:
            ret = method(self, *args, **kwargs)

        return ret
        
    return wrapper


class Inferencer:
    """
    Wraps classifier model to handle classifier inference and opponent stroke
    inference. Uses a semaphore to handle access to model resource

    Models are deployed in TensorRT rather than alternatives such as ORT because
    model gradients are necessary in order to optimize strokes for the AI opponent
    as well as for the gram cam.

    :param classifier_model: instance of classifier model
    """
    def __init__(self, classifier_model: torch.nn.Module):
        self._model = classifier_model
        self._grad_cam = XGradCAM(
            model=classifier_model,
            target_layers=[layer for layer in self._model.conv][0:7], # total 19
            use_cuda=(Settings().device == "cuda")
        )

        self.semaphore = threading.Semaphore(1)


    @async_inference
    def infer_image(self, image: Image):
        input = pil_to_input(image)
        with torch.no_grad():
            logits, _confidences = self._model(input)

        return logits[0].tolist()


    @async_inference
    def infer_image_with_cam(self, image: Image, target_index: int) -> torch.Tensor:
        input = pil_to_input(image)

        targets = [ClassifierOutputTarget(target_index)]
        grayscale_cam = self._grad_cam(input_tensor=input, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        image = pil_rgba_to_rgb(image)
        image_numpy = numpy.asarray(image)
        image_numpy = image_numpy / 255
        grad_cam_image = show_cam_on_image(image_numpy, grayscale_cam, use_rgb=True, image_weight=0.8)

        with torch.no_grad():
            logits, _confidences = self._model(input)

        return logits[0].tolist(), grad_cam_image.tolist()


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
