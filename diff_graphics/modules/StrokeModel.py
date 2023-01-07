from typing import List, Tuple

import torch

from utils.BezierCurve import BezierCurve
from modules import PointGraphic2d, LineGraphic2d, CurveGraphic2d

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
        initial_inputs: List[torch.Tensor],
        max_length: float,
        **path_kwargs,
    ):
        super().__init__()

        # note inputs are optimized
        self.inputs = [
            torch.nn.Parameter(initial_key_point, requires_grad=True)
            for initial_key_point in initial_inputs
        ]
        self.max_length = max_length

        # path rendering
        if len(self.inputs) == 1:
            point_kwargs = {
                kwarg: path_kwargs[kwarg]
                for kwarg in path_kwargs
                if kwarg in ["width", "anti_aliasing_factor"]
            }
            self.graphic = PointGraphic2d(canvas_shape, **point_kwargs)

        if len(self.inputs) == 2:
            line_kwargs = {
                kwarg: path_kwargs[kwarg]
                for kwarg in path_kwargs
                if kwarg in ["width", "anti_aliasing_factor"]
            }
            self.graphic = LineGraphic2d(canvas_shape, **line_kwargs)

        if len(self.inputs) >= 3:
            self.graphic = CurveGraphic2d(canvas_shape, **path_kwargs)


    def parameters(self):
        """
        Future: do this properly, something like making the parameters a module?
        """
        return self.inputs


    def forward(self):
        output_canvas = self.graphic(self.inputs)

        return output_canvas


    def update_graph_width(new_width: float):
        self.graphic.width = new_width


    def update_graph_anti_aliasing_factor(new_factor: float):
        self.graphic.anti_aliasing_factor = new_factor


    def constrain_keypoints(self):
        """
        Make sure this is called within the context torch.no_grad()
        """

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

                self.inputs[1].data = new_endpoint.data

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
        self.inputs[0].clamp_(0.0, 1.0)
        self.inputs[-1].clamp_(0.0, 1.0)
