from typing import List

import torch
import numpy

from helpers import get_uniform_ts, cumulative_sum, bernstein_polynomial


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
