import os
import numpy
from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_tensor
from pytorch_grad_cam import XGradCAM

#GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from train.train_model import Classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Inferencer:
    def __init__(self, model_checkpoint_path):
        self.model = self.load_model(model_checkpoint_path)
        self.cam = XGradCAM(
            model=self.model,
            target_layers=[layer for layer in self.model.conv][0:7], # total 19
            use_cuda=(True if DEVICE == "cuda" else False)
        )

    def load_model(self, model_checkpoint_path):
        model = Classifier()
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE))
        model = model.eval()
        return model

    def _convert_image_to_input(self, image):
        image = image.convert("RGB")
        image = ImageOps.invert(image)
        red_channel = image.split()[0]
        input = to_tensor(red_channel)
        input = torch.reshape(input, (1, 1, 28, 28))

        return input

    def infer_image(self, image):
        input = self._convert_image_to_input(image)
        with torch.no_grad():
            logits, confidences = self.model(input)
        return logits[0].tolist()

    def rgba_to_rgb(self, image):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])

        return background

    def infer_image_with_cam(self, image, target_index):
        input = self._convert_image_to_input(image)

        targets = [ClassifierOutputTarget(target_index)]

        grayscale_cam = self.cam(input_tensor=input, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        image = self.rgba_to_rgb(image)
        image_numpy = numpy.asarray(image)
        image_numpy = image_numpy / 255
        grad_cam_image = show_cam_on_image(image_numpy, grayscale_cam, use_rgb=True, image_weight=0.8)

        with torch.no_grad():
            logits, confidences = self.model(input)
        return logits[0].tolist(), grad_cam_image.tolist()
