from typing import List, Union, Optional

import torch
import numpy

from .BezierCurve import BezierCurve
from .helpers import get_uniform_ts

EPSILON = 0.000001  # autograd has a hard time with 0.0^x


class CurveGraphic2d(torch.nn.Module):
    """
    Samples points from bezier curve(s) and draws them on blank canvas(es)

    :param canvas_shape: shape of output canvas
    :param num_samples: number of samples used to sample bezier curve(s)
    :param max_length: maximum arc length of bezier curve(s)
    """
    def __init__(
        self,
        canvas_shape: List[int],
        num_samples: int = 15,
        max_length: float = 80,
    ):
        super().__init__()
        self.canvas_shape = canvas_shape
        self.num_samples = num_samples
        self.max_length = max_length

        self.max_distance = torch.norm(
            torch.tensor(canvas_shape) - torch.tensor([0.0, 0.0])
        )

        if self.num_samples < 3:
            raise ValueError("num_samples must be greater than 3")


    def forward(
        self,
        inputs: torch.tensor,
        widths: List[float],
        aa_factors: List[float]
    ) -> torch.Tensor:
        """
        Draw normalized keypoints onto canvas

        :param inputs: normalized tensor of keypoints
        :param widths: list of widths for each keypoint
        :param aa_factors: list of anti-aliasing factors for each keypoint
        :return: canvases with drawn curves
        """
        assert len(inputs) == len(widths) == len(aa_factors)

        # prepare canvas and key_points: TODO simplify this
        canvas_shape_tensor = torch.tensor(self.canvas_shape, device=inputs.device)
        key_points = inputs * canvas_shape_tensor[None, None, :]

        # get sample points
        sample_points = self.get_sample_points(key_points)

        # allocate position array
        positions = torch.from_numpy(numpy.array([[
            [y, x]
            for y in range(self.canvas_shape[0])
            for x in range(self.canvas_shape[1])
        ]]))

        # prepare tensors
        positions_repeated = positions.repeat(sample_points.shape[0], sample_points.shape[1], 1)
        sample_points_repeated = torch.repeat_interleave(sample_points, positions.shape[1], dim=1)

        # compute distances between each pixel position and sample points
        distances = torch.norm(positions_repeated - sample_points_repeated, dim=2)
        distances = distances.reshape((sample_points.shape[0], sample_points.shape[1], -1))

        # take the minimum distance for each position
        minimum_distances = torch.min(distances, dim=1).values

        # transform distances into pixel values
        widths = torch.tensor(widths).reshape((-1, 1)).repeat(1, minimum_distances.shape[1])
        aa_factors = torch.tensor(aa_factors).reshape((-1, 1)).repeat(1, minimum_distances.shape[1])

        canvas = torch.zeros(self.canvas_shape, device=inputs.device, dtype=inputs.dtype)
        canvas = minimum_distances / widths
        canvas = canvas + EPSILON
        canvas = canvas ** aa_factors
        canvas = 1 - canvas
        canvas = torch.clamp(canvas, 0.0, 1.0)
        canvas = canvas.reshape((sample_points.shape[0], *self.canvas_shape))

        return canvas


    def get_sample_points(self, key_points: torch.tensor) -> torch.Tensor:
        curves = [
            BezierCurve(key_point_set, num_approximations=self.num_samples)
            for key_point_set in key_points
        ]

        for curve in curves:
            if curve.arc_length > self.max_length:
                curve.truncate(self.max_length / curve.arc_length)

        sample_ts = get_uniform_ts(self.num_samples)

        # TODO: rework
        sample_points = torch.cat([
            torch.cat([
                curve.sample(sample_t)
                for sample_t in sample_ts
            ])
            for curve in curves
        ]).reshape((len(curves), -1, 2))

        return sample_points


    @property
    def device(self):
        return next(self.parameters()).device
    

    @property
    def dtype(self):
        return next(self.parameters()).dtype
