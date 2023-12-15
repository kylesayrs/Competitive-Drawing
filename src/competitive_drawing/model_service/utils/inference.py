from typing import Tuple

import numpy
from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_tensor
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


DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
DEBUG = False


class Inferencer:
    def __init__(self, model_class, state_dict):
        self._model_class = model_class
        self._state_dict = state_dict
        self._model = self.load_model(model_class, state_dict)
        self.grad_cam = XGradCAM(
            model=self._model,
            target_layers=[layer for layer in self._model.conv][0:7], # total 19
            use_cuda=(True if DEVICE == "cuda" else False)
        )
        self.image_size = int(Settings.get("IMAGE_SIZE", 50))


    def load_model(self, model_class, state_dict):
        model = model_class()
        model.load_state_dict(state_dict)
        model = model.eval()
        model = model.to(DEVICE)

        return model


    def convert_image_to_input(self, image: Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = ImageOps.invert(image)
        red_channel = image.split()[0]

        input = to_tensor(red_channel)
        input = torch.reshape(input, (1, 1, self.image_size, self.image_size))
        input = input.to(DEVICE)

        return input


    def infer_image(self, image: Image):
        input = self.convert_image_to_input(image)
        with torch.no_grad():
            logits, confidences = self._model(input)

        return logits[0].tolist()


    def rgba_to_rgb(self, image: Image):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])

        return background


    def infer_image_with_cam(self, image: Image, target_index: int) -> torch.Tensor:
        input = self.convert_image_to_input(image)

        targets = [ClassifierOutputTarget(target_index)]

        grayscale_cam = self.grad_cam(input_tensor=input, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        image = self.rgba_to_rgb(image)
        image_numpy = numpy.asarray(image)
        image_numpy = image_numpy / 255
        grad_cam_image = show_cam_on_image(image_numpy, grayscale_cam, use_rgb=True, image_weight=0.8)

        with torch.no_grad():
            logits, confidences = self._model(input)
        return logits[0].tolist(), grad_cam_image.tolist()


    def infer_stroke(
        self,
        image: Image,
        target_index: int,
        line_width: float,
        max_length: float,
    ):
        base_canvas = self.convert_image_to_input(image)[0][0]

        tmp_target_model = self.load_model(self._model_class, self._state_dict)
        _loss, keypoints = grid_search_stroke(
            base_canvas,
            tmp_target_model,
            target_index,
            torch.optim.Adamax,
            { "lr": 0.03 },
            SearchParameters(
                max_width=line_width * 4,
                min_width=line_width
            ),
            max_length=max_length,
        )
        del tmp_target_model

        curve = BezierCurve(keypoints, num_approximations=20)
        stroke_samples = [
            curve.sample(t).cpu().detach().tolist()
            for t in get_uniform_ts(20)
        ]

        if DEBUG:
            print(f"stroke_samples: {stroke_samples}")

        return stroke_samples


    @property
    def model(self):
        return self._model
