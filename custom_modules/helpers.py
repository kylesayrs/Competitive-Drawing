from typing import List

import cv2
import math
import torch
import numpy
from functools import cache


@cache
def nCr(n: int, r: int):
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


@cache
def bernstein_polynomial(n: int, k: int, t: float):
    return nCr(n, k) * (t ** k) * ((1 - t) ** (n - k))


def get_uniform_ts(num_ts):
    return [t / (num_ts - 1) for t in range(num_ts)]


def cumulative_sum(array):
    return [
        sum(array[:i], 0.0)
        for i in range(len(array) + 1)
    ]


def draw_output_and_target(output_canvas: torch.tensor, target_canvas: torch.tensor):
    assert output_canvas.shape == target_canvas.shape
    image = numpy.zeros((*output_canvas.shape, 3))

    output = output_canvas.detach().numpy()
    target = target_canvas.detach().numpy()

    image[:, :, 0] = cv2.bitwise_and(1.0 - output, target)
    image[:, :, 1] = cv2.bitwise_and(output, target)
    image[:, :, 2] = cv2.bitwise_and(output, 1.0 - target)

    cv2.imshow("output and target", image)
    cv2.waitKey(0)
