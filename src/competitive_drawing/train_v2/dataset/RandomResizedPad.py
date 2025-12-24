from typing import Tuple

import random
import torch
import torchvision.transforms.functional as F


class RandomResizedPad(object):
    """
    Randomly crop an image then pad it to its original size
    """
    def __init__(
        self,
        min_scale: int = 0.2,
        max_scale: int = 1.0,
        mode: str = "constant",
        value: int = 0,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = mode
        self.value = value


    def rand(self, a, b):
        return (b - a) * random.random() + a


    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """
        sample.shape = (..., H, W)
        """
        sample_shape = sample.shape

        scale = self.rand(self.min_scale, self.max_scale)
        resize_shape = (round(sample_shape[-2] * scale), round(sample_shape[-1] * scale))
        resized = F.resize(sample, resize_shape)

        left = self.rand(0, sample_shape[-1] - resize_shape[-1])
        left = round(left)
        right = sample_shape[-1] - left - resized.shape[-1]

        top = self.rand(0, sample_shape[-2] - resized.shape[-2])
        top = round(top)
        bottom = sample_shape[-2] - top - resized.shape[-2]

        pad = (left, right, top, bottom)

        output = torch.nn.functional.pad(resized, pad, mode=self.mode, value=self.value)

        return output


if __name__ == "__main__":
    from torchvision.io import read_image, write_png, ImageReadMode

    file_path = "src/competitive_drawing/web_app/static/assets/logo.png"
    image = read_image(file_path, mode=ImageReadMode.GRAY)
    transform = RandomResizedPad()

    new_image = transform(image)
    write_png(new_image, "random_resize_padded_image.png")
