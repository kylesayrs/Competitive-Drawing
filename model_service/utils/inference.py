import os
import numpy
from PIL import Image, ImageOps
from dotenv import load_dotenv

import torch
from torchvision.transforms.functional import to_tensor
from pytorch_grad_cam import XGradCAM

#GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

load_dotenv(".env")
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Inferencer:
    def __init__(self, model_class, state_dict):
        self.model = self.load_model(model_class, state_dict)
        self.cam = XGradCAM(
            model=self.model,
            target_layers=[layer for layer in self.model.conv][0:7], # total 19
            use_cuda=(True if DEVICE == "cuda" else False)
        )
        self.image_size = int(os.environ.get("IMAGE_SIZE", 50))


    def load_model(self, model_class, state_dict):
        model = model_class()
        model.load_state_dict(state_dict)
        model = model.eval()
        model = model.to(DEVICE)

        return model


    def _convert_image_to_input(self, image: Image):
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
        input = self._convert_image_to_input(image)
        with torch.no_grad():
            logits, confidences = self.model(input)

        return logits[0].tolist()


    def rgba_to_rgb(self, image: Image):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])

        return background


    def infer_image_with_cam(self, image: Image, target_index: int):
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
