from typing import List, Union, Optional

import torch
import numpy

from .BezierCurve import BezierCurve
from .helpers import get_uniform_ts

EPSILON = 0.000001  # autograd has a hard time with 0.0^x


class CurveGraphic2d(torch.nn.Module):
    """

    """

    def __init__(
        self,
        canvas_shape: List[int],
        num_samples: int = 15,
        max_length: float = 80,
        device: Union[torch.device, str] = "cpu",
        dtype: Union[torch.dtype, str] = torch.float32
    ):
        super().__init__()
        self.canvas_shape = canvas_shape
        self.num_samples = num_samples
        self.max_length = max_length

        self._device = device
        self._dtype = dtype
        self.max_distance = torch.norm(
            torch.tensor(canvas_shape) - torch.tensor([0.0, 0.0])
        )

        if self.num_samples < 3:
            raise ValueError("num_samples must be greater than 3")


    def forward(self, inputs: List[torch.tensor], widths: List[float], aa_factors: List[float]):
        # prepare canvas and key_points: TODO simplify this
        canvas_shape_tensor = torch.tensor(self.canvas_shape, device=self._device)
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

        canvas = torch.zeros(self.canvas_shape, device=self._device, dtype=self._dtype)
        canvas = minimum_distances / widths
        canvas = canvas + EPSILON
        canvas = canvas ** aa_factors
        canvas = 1 - canvas
        canvas = torch.clamp(canvas, 0.0, 1.0)
        canvas = canvas.reshape((sample_points.shape[0], *self.canvas_shape))

        return canvas


    def get_sample_points(self, key_points: torch.tensor):
        curves = [
            BezierCurve(
                key_point_set,
                num_approximations=self.num_samples  # technically could be anything
            )
            for key_point_set in key_points
        ]

        for curve in curves:
            if curve.arc_length > self.max_length:
                curve.truncate(self.max_length / curve.arc_length)

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


    def to(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[Union[torch.dtype, str]] = None
    ):
        super().to(device, dtype)

        if device is not None:
            self._device = device

        if dtype is not None:
            self._dtype = dtype

        return self
