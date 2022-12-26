from typing import List

import time
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
