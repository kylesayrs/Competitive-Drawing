from typing import List

import time
import math
import torch
from functools import cache


@cache
def nCr(n: int, r: int):
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


@cache
def bernstein_polynomial(n: int, k: int, t: float):
    return nCr(n, k) * (t ** k) * (1 - t) ** (n - k)


def bezier_curve(key_points: List[torch.tensor], t: float):
    return sum([
        key_point * bernstein_polynomial(len(key_points), key_point_i, t)
        for key_point_i, key_point in enumerate(key_points)
    ])


def get_line_precomputations(key_points: List[torch.tensor]):
    p1_minus_p0 = key_points[1] - key_points[0]
    len_line_segment_squared = (
        (key_points[1][0] - key_points[0][0]) ** 2 +
        (key_points[1][1] - key_points[0][1]) ** 2
    )

    return p1_minus_p0, len_line_segment_squared


def get_bezier_curve_precomputations(key_points: List[torch.tensor], num_path_samples: int):
    if True:#self.path_sample_method == "uniform":
        # note: ideally we do uniform sampling with respect to distance along the line, not t
        # this can be done with the LUT table method
        sample_ts = [i / num_path_samples for i in range(0, num_path_samples)]

    sample_points = [
        bezier_curve(key_points, sample_t)
        for sample_t in sample_ts
    ]

    return sample_points
