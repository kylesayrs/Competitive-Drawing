from typing import List, Tuple

import torch

from competitive_drawing.diff_graphics.utils.BezierCurve import BezierCurve
from .PointGraphic2d import PointGraphic2d
from .LineGraphic2d import LineGraphic2d
from .CurveGraphic2d import CurveGraphic2d

EPSILON = 0.001


class StrokeModel(torch.nn.Module):
    """
    Given a canvas, score model, and maximum path length,
    this model finds an optimal path that maximizes the
    score with the given contraints

    The first technique places the path at random positions
    on the canvas, optimizes each, then picks the optimized
    path with the highest score

    The second technique starts with a path with a large
    aliasing width, then slowly decreases that width as the
    path is optimized, converging on a global solution
    """

    def __init__(
        self,
        canvas_shape: Tuple[float, float],
        initial_inputs: torch.Tensor, # shape = (num_batches, num_key_points, 2)
        max_length: float,
        widths: List[float],
        aa_factors: List[float],
        **curve_kwargs,
    ):
        super().__init__()

        # note inputs are optimized
        self.inputs = torch.nn.Parameter(initial_inputs, requires_grad=True)
        self.max_length = max_length
        self.widths = widths
        self.aa_factors = aa_factors
        self._device = "cpu"

        self.graphic = CurveGraphic2d(canvas_shape, **curve_kwargs)


    def forward(self):
        output_canvas = self.graphic(self.inputs, self.widths, self.aa_factors)

        return output_canvas


    def update_width_and_anti_aliasing(
        self,
        scores: List[float],
        max_width: float,
        min_width: float,
        max_aa: float,
        min_aa: float,
    ):
        with torch.no_grad():
            self.widths = [
                (1.0 - score) * max_width + (score * min_width)
                for score in scores
            ]
            self.aa_factors = [
                (1.0 - score) * max_aa + (score * min_aa)
                for score in scores
            ]


    def constrain_keypoints(self):
        """
        Make sure this is called within the context torch.no_grad()
        """
        with torch.no_grad():
            # enforce maximum length
            canvas_shape_tensor = torch.tensor(self.graphic.canvas_shape, device=self._device)
            key_points = [input * canvas_shape_tensor for input in self.inputs]
            curves = [
                BezierCurve(
                    key_point_set,
                    num_approximations=self.graphic.num_samples  # technically could be anything
                )
                for key_point_set in key_points
            ]

            for curve_i, curve in enumerate(curves):
                if curve.arc_length > self.max_length:
                    curve.truncate(self.max_length / curve.arc_length)

                    new_key_points = curve.key_points
                    new_inputs = [key_point / canvas_shape_tensor for key_point in new_key_points]
                    for input_i, new_input in enumerate(new_inputs):
                        self.inputs.data[curve_i][input_i] = new_input.data


            # clamp endpoints
            self.inputs[:, 0].clamp_(0.0, 1.0)
            self.inputs[:, -1].clamp_(0.0, 1.0)


    def to(self, device):
        super().to(device)
        self._device = device
        for input in self.inputs:
            input.data = input.data.to(device)

        self.graphic = self.graphic.to(device)

        return self
