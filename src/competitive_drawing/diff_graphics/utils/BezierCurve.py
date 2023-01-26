from typing import List

import torch
import numpy

from .helpers import get_uniform_ts, bernstein_polynomial


class BezierCurve():
    def __init__(
        self,
        key_points: List[torch.tensor],
        num_approximations: int = 20
    ):
        self.key_points = key_points
        self.num_approximations = num_approximations
        self._device = key_points[0].data.device

        self._approx_ts = get_uniform_ts(num_approximations)
        self._approx_points = [
            self._sample_directly(approx_t)
            for approx_t in self._approx_ts
        ]
        self._approx_lengths = self._get_cumulative_distances()
        self._approx_lengths_normalized = self._get_cumulative_distances_normalized()


    def _get_cumulative_distances(self):
        return torch.cumsum(
            torch.cat([
                torch.norm(self._approx_points[i] - self._approx_points[i + 1], keepdim=True)
                for i in range(len(self._approx_points) - 1)
            ]),
            dim=0
        )


    def _get_cumulative_distances_normalized(self):
        return torch.tensor([
            cumulative_sum / self.arc_length
            for cumulative_sum in self._approx_lengths
        ], device=self._device)


    def _sample_directly(self, t: float):
        """
        TODO: batch samples to rewrite as a matrix multiplication
        """
        return sum([
            key_point * bernstein_polynomial(len(self.key_points) - 1, key_point_i, t)
            for key_point_i, key_point in enumerate(self.key_points)
        ])


    def sample(self, t: float):
        return self._sample_directly(t)


    @property
    def arc_length(self):
        return float(self._approx_lengths[-1])


    def truncate(self, normed_t: float):
        """
        Subdivision algorithm
        """
        # randomly decide which endpoint to truncate
        truncate_left = torch.rand(1)[0] > 0.5
        if truncate_left:
            self.key_points = list(reversed(self.key_points))

        with torch.no_grad():
            right_index = torch.searchsorted(self._approx_lengths_normalized, normed_t, side="right")
            left_index = right_index - 1

            if right_index <= 0:
                real_t = self._approx_ts[0]

            if left_index >= self.num_approximations - 1:
                real_t = self._approx_ts[-1]

            else:
                lerp_t = (
                    (normed_t - self._approx_lengths_normalized[left_index]) /
                    (
                        self._approx_lengths_normalized[right_index] -
                        self._approx_lengths_normalized[left_index]
                    )
                )

                real_t = torch.lerp(
                    torch.tensor(self._approx_ts[left_index], device=self.key_points[0].device),
                    torch.tensor(self._approx_ts[right_index], device=self.key_points[0].device),
                    lerp_t
                )

            degree_key_points = [self.key_points]  # first degree
            for prev_degree in range(len(self.key_points) - 1):

                key_points = [
                    torch.lerp(
                        degree_key_points[prev_degree][kp_i],
                        degree_key_points[prev_degree][kp_i + 1],
                        real_t
                    )
                    for kp_i in range(len(degree_key_points[prev_degree]) - 1)
                ]

                degree_key_points.append(key_points)

            new_key_points = [
                degree_key_points[degree][0]
                for degree in range(len(self.key_points))
            ]

        if truncate_left:
            self.key_points = list(reversed(self.key_points))

        self.__init__(
            new_key_points,
            num_approximations=self.num_approximations
        )
