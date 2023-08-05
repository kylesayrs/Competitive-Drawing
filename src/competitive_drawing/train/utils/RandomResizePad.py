from typing import Tuple

import numpy
import torch
import torchvision.transforms.functional as F


class RandomResizePad(object):
    """
    Randomly crop an image then pad it to its original size
    """
    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.2, 1.0),
        mode: str = "constant",
        value: int = 0,
    ):
        self.size = size
        self.scale = scale
        self.mode = mode
        self.value = value


    def rand(self, a, b):
        return (b - a) * numpy.random.rand(1)[0] + a


    def __call__(self, sample: torch.tensor):
        """
        sample.shape = (B, C, W, H)
        """
        scale = self.rand(*self.scale)
        resize_shape = (round(sample.shape[2] * scale), round(sample.shape[3] * scale))
        resized = F.resize(sample, resize_shape)

        left = self.rand(0, self.size[0] - resized.shape[2])
        left = round(left)
        right = self.size[0] - left - resized.shape[2]

        top = self.rand(0, self.size[1] - resized.shape[3])
        top = round(top)
        bottom = self.size[1] - top - resized.shape[3]

        pad = (left, right, top, bottom)

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
