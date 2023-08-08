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
        scale: Tuple[float, float] = (0.2, 1.0),
        mode: str = "constant",
        value: int = 0,
    ):
        self.scale = scale
        self.mode = mode
        self.value = value


    def rand(self, a, b):
        return (b - a) * numpy.random.rand(1)[0] + a


    def __call__(self, sample: torch.tensor):
        """
        sample.shape = (C, W, H)
        """
        sample_size = sample.shape[1:3]

        scale = self.rand(*self.scale)
        resize_shape = (round(sample_size[0] * scale), round(sample_size[1] * scale))
        resized = F.resize(sample, resize_shape)

        left = self.rand(0, sample_size[0] - resized.shape[1])
        left = round(left)
        right = sample_size[0] - left - resized.shape[1]

        top = self.rand(0, sample_size[1] - resized.shape[2])
        top = round(top)
        bottom = sample_size[1] - top - resized.shape[2]

        pad = (left, right, top, bottom)

        output = torch.nn.functional.pad(resized, pad, mode=self.mode, value=self.value)

        return output


if __name__ == "__main__":
    import cv2

    image = cv2.imread("../../static/assets/logo.png", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = torch.tensor([image])
    random_resize_pad = RandomResizePad()

    new_image = random_resize_pad(image)
    print(new_image.shape)
    cv2.imwrite("./new_image.png", new_image.numpy()[0])
