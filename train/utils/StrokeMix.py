from typing import Tuple

import numpy
import torch
import torchvision.transforms.functional as F


class StrokeMix(object):
    def __init__(
        self,
        scale: Tuple[float, float],
        mode: str = "add",
    ):
        self.scale = scale
        self.mode = mode

        if self.mode not in ["add", "replace"]:
            raise ValueError("StrokeMix mode must be either add or replace")


    def rand(self, a: float, b: float):
        return (b - a) * numpy.random.rand(1)[0] + a


    def __call__(self, sample: torch.tensor):
        """
        sample.shape = (B, C, W, H)
        """
        crop_shape = (
            round(self.rand(self.scale[0], self.scale[1]) * sample.shape[2]),
            round(self.rand(self.scale[0], self.scale[1]) * sample.shape[3])
        )

        crop_shape
        torchvision.transforms.functional.crop(img: Tensor, top: int, left: int, height: int, width: int)

        output = torch.nn.functional.pad(resized, pad, mode=self.mode, value=self.value)

        return output


if __name__ == "__main__":
    import cv2

    image = cv2.imread("../../static/assets/logo.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = torch.tensor([image])
    random_resize_pad = RandomResizePad((50, 50))

    new_image = random_resize_pad(image)
    print(new_image.shape)
    cv2.imwrite("./new_image.png", new_image.numpy()[0])
