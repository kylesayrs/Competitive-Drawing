from typing import List

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


def make_hooked_optimizer(optimizer_class, hook, *optimizer_args, **optimizer_kwargs):
    class HookedOptimizer(optimizer_class):
        def step(self, closure=None):
            super().step(closure)
            with torch.no_grad():
                hook()

    return HookedOptimizer(*optimizer_args, **optimizer_kwargs)


def draw_output_and_target(output_canvas: torch.tensor, target_canvas: torch.tensor):
    assert output_canvas.shape == target_canvas.shape
    image = numpy.zeros((*output_canvas.shape, 3))

    output = output_canvas.cpu().detach().numpy()
    target = target_canvas.cpu().detach().numpy()

    image[:, :, 0] = (1.0 - output) * target
    image[:, :, 1] = output * target * 2
    image[:, :, 2] = output * (1.0 - target)

    return image


def torch_search(array, value):
    return torch.argmin(torch.abs(array - value))
