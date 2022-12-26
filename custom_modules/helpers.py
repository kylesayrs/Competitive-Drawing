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


def get_line_precomputations(key_points: List[torch.tensor]):
    p1_minus_p0 = key_points[1] - key_points[0]
    len_line_segment_squared = (
        (key_points[1][0] - key_points[0][0]) ** 2 +
        (key_points[1][1] - key_points[0][1]) ** 2
    )

    return p1_minus_p0, len_line_segment_squared


def get_uniform_ts(num_ts):
    return [t / (num_ts - 1) for t in range(num_ts)]


def cumulative_sum(array):
    return [
        sum(array[:i], 0.0)
        for i in range(len(array) + 1)
    ]


class BezierCurve():
    def __init__(
        self,
        key_points: List[torch.tensor],
        sample_method: str = "uniform",
        num_approximations: int = 20
    ):
        self.key_points = key_points
        self.sample_method = sample_method

        self._approx_ts = get_uniform_ts(num_approximations)
        self._approx_points = [
            self._sample_directly(approx_t)
            for approx_t in self._approx_ts
        ]
        self._approx_lengths = self._get_cumulative_distances()
        self._approx_lengths_normalized = self._get_cumulative_distances_normalized()


    def _get_cumulative_distances(self):
        with torch.no_grad():
            return cumulative_sum([
                torch.norm(self._approx_points[i] - self._approx_points[i + 1])
                for i in range(len(self._approx_points) - 1)
            ])


    def _get_cumulative_distances_normalized(self):
        return [
            cumulative_sum / self.arc_length()
            for cumulative_sum in self._approx_lengths
        ]


    def _sample_directly(self, t: float):
        return sum([
            key_point * bernstein_polynomial(len(self.key_points) - 1, key_point_i, t)
            for key_point_i, key_point in enumerate(self.key_points)
        ])


    def _sample_from_approximations(self, t: float):
        right_index = numpy.searchsorted(self._approx_lengths_normalized, t, side="right")
        left_index = right_index - 1

        # edge cases
        if right_index <= 0:
            return self._approx_points[0]
        if left_index >= len(self._approx_ts) - 1:
            return self._approx_points[-1]

        # sample by lerping
        lerp_t = (
            (t - self._approx_lengths_normalized[left_index]) /
            (
                self._approx_lengths_normalized[right_index] -
                self._approx_lengths_normalized[left_index]
            )
        )

        if lerp_t == 0.0:
            return self._approx_points[left_index]
        if lerp_t == 1.0:
            return self._approx_points[right_index]

        return torch.lerp(
            self._approx_points[left_index],
            self._approx_points[right_index],
            lerp_t
        )


    def sample(self, t: float):
        if self.sample_method == "uniform_t":
            return self._sample_directly(t)

        elif self.sample_method == "uniform":
            return self._sample_from_approximations(t)

        else:
            raise ValueError(f"Unknown sampling method {self.sample_method}")


    def arc_length(self):
        return float(self._approx_lengths[-1])



def get_bezier_curve_precomputations(
    key_points: List[torch.tensor],
    num_path_samples: int,
    sample_method: str,
):
    bezier_curve = BezierCurve(
        key_points,
        sample_method=sample_method,
        num_approximations=num_path_samples
    )

    sample_ts = get_uniform_ts(num_path_samples)

    sample_points = [bezier_curve.sample(sample_t) for sample_t in sample_ts]

    return sample_points
