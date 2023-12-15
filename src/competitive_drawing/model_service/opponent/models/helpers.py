from typing import List

import math
import torch
import numpy
from functools import cache


@cache
def nCr(n: int, r: int) -> int:
    """
    N choose R

    :param n: number of elements to choose from
    :param r: number of elements being chosen
    :return: number of possible r choices of n elements
    """
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


@cache
def bernstein_polynomial(n: int, k: int, t: float) -> int:
    """
    Given n-many points, finds the coefficient for the kth point for a
    bezier curve parameterized by t

    For example, for points { p1, p2, p3, p4 },
    P(t) = 
        p1 * bernstein_polynomial(4, 1, t) +
        p2 * bernstein_polynomial(4, 2, t) +
        p3 * bernstein_polynomial(4, 3, t) +
        p4 * bernstein_polynomial(4, 4, t)

    :param n: number of points
    :param k: point index whose coefficient is being calculated
    :param t: bezier curve parameterization
    :return: coefficent for point k
    """
    return nCr(n, k) * (t ** k) * ((1 - t) ** (n - k))


def get_uniform_ts(num_ts: int) -> List[float]:
    """
    Get a list of points which evenly divide the interval [0, 1)

    :param num_ts: number of uniform segments
    :return: split points
    """
    return [t / (num_ts - 1) for t in range(num_ts)]


def torch_fuzzy_search(array: torch.Tensor, value: float) -> int:
    """
    Finds the index in array closest to value

    :param array: array to search
    :param value: target value to search for
    :return: index of array whose value is closest to value
    """
    return torch.argmin(torch.abs(array - value))
