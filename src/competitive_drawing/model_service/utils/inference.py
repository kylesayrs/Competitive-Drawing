from typing import Tuple

import numpy
from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_tensor
from pytorch_grad_cam import XGradCAM
#GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from competitive_drawing import Settings
from competitive_drawing.diff_graphics.search import grid_search_stroke
from competitive_drawing.diff_graphics.utils import BezierCurve, get_uniform_ts


DEVICE = (
    #"mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

class Inferencer:
    def __init__(self, model_class, state_dict):
        self._model = self.load_model(model_class, state_dict)
        self.cam = XGradCAM(
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


    def convert_image_to_input(self, image: Image):
        image = image.convert("RGB")
        image = ImageOps.invert(image)
        red_channel = image.split()[0]

        """
        import cv2
        cv2.imwrite("/Users/poketopa/Desktop/image.png", numpy.array(red_channel))
        """

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


    def infer_image_with_cam(self, image: Image, target_index: int):
        input = self.convert_image_to_input(image)

        targets = [ClassifierOutputTarget(target_index)]

        grayscale_cam = self.cam(input_tensor=input, targets=targets)
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

        loss, keypoints = grid_search_stroke(
            base_canvas,
            (3, 3),
            self.model,
            target_index,
            torch.optim.Adamax,
            { "lr": 0.02 },
            max_width=line_width * 8,
            min_width=line_width * 2,
            max_aa=0.35,
            min_aa=0.9,
            max_steps=20,#100,
            save_best=True,
            draw_output=False,
            max_length=max_length,
        )

        curve = BezierCurve(
            keypoints,
            sample_method="uniform",
            num_approximations=20
        )
        stroke_samples = [
            curve.sample(t).cpu().detach().tolist()
            for t in get_uniform_ts(20)
        ]

        print(f"stroke_samples: {stroke_samples}")
        return stroke_samples


    @property
    def model(self):
        return self._model
