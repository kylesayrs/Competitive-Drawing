from typing import List

import torch
import numpy

from .helpers import get_uniform_ts, bernstein_polynomial


class BezierCurve():
    def __init__(
        self,
        key_points: torch.tensor, # shape = (num_batches, num_key_points, 2)
        sample_method: str = "uniform",
        num_approximations: int = 20
    ):
        self.key_points = key_points
        self.sample_method = sample_method
        self.num_approximations = num_approximations
        self._device = key_points.data.device

        self._approx_ts = get_uniform_ts(num_approximations)
        self._approx_points = torch.cat([
            self._sample_directly(approx_t)
            for approx_t in self._approx_ts
        ])
        self._approx_lengths = self._get_cumulative_distances(self._approx_points)
        self._approx_lengths_normalized = self._approx_lengths / self.arc_lengths


    def _get_cumulative_distances(self, approx_points):
        with torch.no_grad():
            distances = torch.cat([torch.zeros(approx_points.shape[1], 1)] + [
                torch.norm(approx_points[i] - approx_points[i + 1], dim=1, keepdim=True)
                for i in range(len(approx_points) - 1)
            ])
            return torch.cumsum(distances, dim=0)


    def _sample_directly(self, t: float):
        coefficients = torch.tensor([
            bernstein_polynomial(self.key_points.shape[1] - 1, key_point_i, t)
            for key_point_i in range(self.key_points.shape[1])
        ])

        return torch.sum(self.key_points * coefficients[None, :, None], dim=1, keepdim=True)


    def _sample_from_approximations(self, t: float):
        right_index = torch.searchsorted(self._approx_lengths_normalized, t, side="right")
        left_index = right_index - 1

        # edge cases
        if right_index <= 0:
            return self._approx_points[0]
        if left_index >= self.num_approximations - 1:
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
        if True:#self.sample_method == "uniform_t":
            return self._sample_directly(t)

        elif self.sample_method == "uniform":
            return self._sample_from_approximations(t)

        else:
            raise ValueError(f"Unknown sampling method {self.sample_method}")


    @property
    def arc_lengths(self):
        return self._approx_lengths[-1]


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
            sample_method=self.sample_method,
            num_approximations=self.num_approximations
        )
