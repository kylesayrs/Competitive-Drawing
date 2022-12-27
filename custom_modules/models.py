from typing import List, Tuple, Optional

import torch

from BezierCurve import BezierCurve
from PointGraphic2d import PointGraphic2d
from LineGraphic2d import LineGraphic2d
from CurveGraphic2d import CurveGraphic2d


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
        initial_inputs: List[Tuple[float, float]],
        max_length: float,
        **path_kwargs,
    ):
        super().__init__()

        # note inputs are optimized
        self.inputs = [
            torch.nn.Parameter(torch.tensor(initial_key_point), requires_grad=True)
            for initial_key_point in initial_inputs
        ]
        self.max_length = max_length

        # path rendering
        if len(self.inputs) == 1:
            self.graphic = PointGraphic2d(canvas_shape, **path_kwargs)

        if len(self.inputs) == 2:
            self.graphic = LineGraphic2d(canvas_shape, **path_kwargs)

        if len(self.inputs) >= 3:
            self.graphic = CurveGraphic2d(canvas_shape, **path_kwargs)


    def parameters(self):
        """
        Future: do this properly, something like making the parameters a module?
        """
        model_parameters = super().parameters()
        return list(model_parameters) + self.inputs


    def forward(self):
        output_canvas = self.graphic(self.inputs)

        return output_canvas


    def update_graph_width(new_width: float):
        self.graphic.width = new_width


    def update_graph_anti_aliasing_factor(new_factor: float):
        self.graphic.anti_aliasing_factor = new_factor


    def constrain_graphic(self):
        with torch.no_grad():
            # enforce maximum length
            canvas_shape_tensor = torch.tensor(self.graphic.canvas_shape)
            key_points = [input * canvas_shape_tensor for input in self.inputs]

            if len(self.inputs) == 1:
                return  # points have no length

            if len(self.inputs) == 2:
                line_length = torch.norm(key_points[1] - key_points[0])
                if line_length > self.max_length:
                    new_endpoint = torch.lerp(
                        key_points[0],
                        key_points[1],
                        self.max_length / line_length
                    )

                new_input = new_endpoint / canvas_shape_tensor
                self.inputs[1] = torch.nn.Parameter(new_input, requires_grad=True)

            if len(self.inputs) >= 3:
                curve = BezierCurve(
                    key_points,
                    sample_method="uniform",
                    num_approximations=self.graphic.num_samples
                )
                arc_length = curve.arc_length()
                if arc_length > self.max_length:
                    curve.truncate(self.max_length / arc_length)

                    new_key_points = curve.key_points
                    new_inputs = [key_point / canvas_shape_tensor for key_point in new_key_points]
                    for input, new_input in zip(self.inputs, new_inputs):
                        input.data = new_input.data

            # clamp endpoints
            self.inputs[0].clamp_(0, 1)
            self.inputs[-1].clamp_(0, 1)


class StrokeScoreModel(StrokeModel):
    def __init__(
        self,
        score_model: torch.nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **path_kwargs)

        # freeze score model
        self.score_model = score_model
        for param in self.score_model.parameters():
            param.requires_grad = False


    def forward(self):
        output_canvas = super().forward()

        return self.score_model(output_canvas)
