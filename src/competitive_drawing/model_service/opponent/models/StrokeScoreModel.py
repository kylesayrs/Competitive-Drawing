from typing import List, Tuple

import torch

from .CurveGraphic2d import CurveGraphic2d
from .BezierCurve import BezierCurve


class StrokeScoreModel(torch.nn.Module):
    """
    Given a canvas, score model, and maximum stroke length,
    finds the curve which maximizes the score within the given contraints

    One technique might be to place curves at random positions on the canvas,
    optimize each, then pick the optimized path with the highest score

    Another approach starts with a randomly initialized curve with a large
    aliasing width, then slowly decreases that width as the ath is optimized,
    converging on a global solution

    :param base_canvas: canvas upon which curves are drawn
    :param initial_inputs: 
    """
    def __init__(
        self,
        base_canvas: torch.Tensor,
        initial_inputs: List[torch.Tensor],
        score_model: torch.nn.Module,
        target_index: int,
        max_length: float,
        widths: List[float],
        aa_factors: List[float],
        **curve_kwargs
    ):
        super().__init__()

        self.base_canvas = base_canvas.to(torch.float32)
        self.score_model = score_model.float()
        self.score_model.eval()
        self.target_index = target_index

        # note inputs are optimized
        self.inputs = torch.nn.Parameter(initial_inputs, requires_grad=True)
        self.max_length = max_length
        self.widths = widths
        self.aa_factors = aa_factors
        self._device = "cpu"

        self.graphic = CurveGraphic2d(base_canvas.shape, **curve_kwargs)

        # freeze score model
        for param in self.score_model.parameters():
            param.requires_grad = False


    def forward(self):
        graphic = self.graphic(self.inputs, self.widths, self.aa_factors)
        canvas_with_graphic = self.base_canvas + graphic
        canvas_with_graphic = torch.reshape(canvas_with_graphic.to(torch.float32), (-1, 1, *self.base_canvas.shape))
        logits, scores = self.score_model(canvas_with_graphic)

        target_scores = scores[:, self.target_index]

        return canvas_with_graphic, target_scores
    

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
        self.base_canvas = self.base_canvas.to(device)
        self.score_model = self.score_model.to(device)

        return self
