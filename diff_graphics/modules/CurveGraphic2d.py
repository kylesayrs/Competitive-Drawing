from typing import List, Optional

import torch

from utils.BezierCurve import BezierCurve
from utils.helpers import get_uniform_ts

EPSILON = 0.000001  # autograd has a hard time with 0.0^x


class CurveGraphic2d(torch.nn.Module):
    """

    """

    def __init__(
        self,
        canvas_shape: List[int],
        width: float = 1.0,
        anti_aliasing_factor: str = 1.0,
        num_samples: int = 10,
        sample_method: str = "uniform"
    ):
        super().__init__()
        self.canvas_shape = canvas_shape
        self.width = width
        self.anti_aliasing_factor = anti_aliasing_factor
        self.num_samples = num_samples
        self.sample_method = sample_method

        self.max_distance = torch.norm(
            torch.tensor(canvas_shape) - torch.tensor([0.0, 0.0])
        )

        if self.num_samples < 3:
            raise ValueError("num_samples must be greater than 3")

        if self.sample_method not in ["uniform", "uniform_t", "stochastic"]:
            raise ValueError(
                "sample_method must be 'uniform', 'uniform_t', or 'stochastic'"
            )


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

        return min([torch.norm(p - sample_point) for sample_point in sample_points])


    def forward(self, inputs: List[torch.tensor]):
        # prepare canvas and key_points
        canvas = torch.zeros(self.canvas_shape)
        canvas_shape_tensor = torch.tensor(self.canvas_shape)
        key_points = [input * canvas_shape_tensor for input in inputs]

        # precompute some values
        b_curve_precomputations = self.get_bezier_curve_precomputations(key_points)

        # torch doesn't implement a map function, so a single thread will do
        for y in range(0, canvas.shape[0]):
            for x in range(0, canvas.shape[1]):
                p = torch.tensor([y, x], dtype=torch.float32)

                distance = self.sample_distance_to_path(
                    p, key_points,
                    b_curve_precomputations=b_curve_precomputations
                )

                if distance < self.width:
                    canvas[y, x] = 1 - (distance / self.width + EPSILON) ** self.anti_aliasing_factor

        return canvas

    def get_bezier_curve_precomputations(self, key_points: List[torch.tensor]):
        bezier_curve = BezierCurve(
            key_points,
            sample_method=self.sample_method,
            num_approximations=self.num_samples  # technically could be anything
        )

        sample_ts = get_uniform_ts(self.num_samples)

        sample_points = [bezier_curve.sample(sample_t) for sample_t in sample_ts]

        return sample_points
