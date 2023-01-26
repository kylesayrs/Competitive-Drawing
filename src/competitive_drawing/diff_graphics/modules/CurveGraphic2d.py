from typing import List, Optional

import torch

from competitive_drawing.diff_graphics.utils.BezierCurve import BezierCurve
from competitive_drawing.diff_graphics.utils.helpers import get_uniform_ts

EPSILON = 0.000001  # autograd has a hard time with 0.0^x


class CurveGraphic2d(torch.nn.Module):
    """

    """

    def __init__(
        self,
        canvas_shape: List[int],
        num_samples: int = 15,
    ):
        super().__init__()
        self.canvas_shape = canvas_shape
        self.num_samples = num_samples

        self._device = "cpu"
        self._canvas = torch.zeros(self.canvas_shape)
        self.max_distance = torch.norm(
            torch.tensor(canvas_shape) - torch.tensor([0.0, 0.0])
        )

        if self.num_samples < 3:
            raise ValueError("num_samples must be greater than 3")


    def sample_distance_to_path(
        self,
        p: torch.tensor,
        key_points: List[torch.tensor],
        b_curve_precomputations: Optional[List[torch.tensor]] = None,
    ):
        """
        """
        sample_points = (
            self.get_bezier_curve_precomputations(key_points)
            if b_curve_precomputations is None
            else b_curve_precomputations
        )

        distances = torch.cat([
            torch.norm(p - curve_sample_points, dim=1)
            for curve_sample_points in sample_points
        ]).reshape((sample_points.shape[0], -1))

        return torch.min(distances, dim=1).values


    def forward(self, inputs: List[torch.tensor], widths: List[float], aa_factors: List[float]):
        # prepare canvas and key_points
        canvas = self._canvas.repeat(inputs.shape[0], 1, 1)
        canvas_shape_tensor = torch.tensor(self.canvas_shape, device=self._device)
        key_points = inputs * canvas_shape_tensor[None, None, :]

        # precompute some values
        b_curve_precomputations = self.get_bezier_curve_precomputations(key_points)

        # torch doesn't implement a map function, so a single thread will do
        for y in range(0, canvas.shape[1]):
            for x in range(0, canvas.shape[2]):
                p = torch.tensor([y, x], dtype=torch.float32, device=self._device)

                distances = self.sample_distance_to_path(
                    p, key_points,
                    b_curve_precomputations=b_curve_precomputations
                )

                for canvas_i in range(len(canvas)):
                    if distances[canvas_i] < widths[canvas_i]:
                        canvas[canvas_i, y, x] = 1 - (distances[canvas_i] / widths[canvas_i] + EPSILON) ** aa_factors[canvas_i]

        return canvas


    def get_bezier_curve_precomputations(self, key_points: torch.tensor):
        curves = [
            BezierCurve(
                key_point_set,
                num_approximations=self.num_samples  # technically could be anything
            )
            for key_point_set in key_points
        ]

        sample_ts = get_uniform_ts(self.num_samples)

        # jank af
        sample_points = torch.cat([
            torch.cat([
                curve.sample(sample_t)
                for sample_t in sample_ts
            ])
            for curve in curves
        ]).reshape((len(curves), -1, 2))

        return sample_points


    def to(self, device):
        super().to(device)
        self._device = device
        self._canvas = self._canvas.to(device)

        return self
