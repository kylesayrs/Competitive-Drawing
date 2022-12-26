import torch

from PointGraphic2d import PointGraphic2d
from LineGraphic2d import LineGraphic2d
from CurveGraphic2d import CurveGraphic2d

class OpponentModel(torch.nn.Module):
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
        canvas_shape,
        num_key_points: int = 4,
        **path_kwargs,
    ):
        super(OpponentModel, self).__init__()

        # key points to optimize
        #""" random key_points
        self.key_points = [
            torch.nn.Parameter(torch.rand(2), requires_grad=True)
            for _ in range(num_key_points)
        ]
        #"""
        """
        self.key_points = [
            torch.nn.Parameter(torch.tensor([0.1, 0.1]), requires_grad=True),
            torch.nn.Parameter(torch.tensor([0.9, 0.2]), requires_grad=True),
            torch.nn.Parameter(torch.tensor([0.0, 0.8]), requires_grad=True),
            torch.nn.Parameter(torch.tensor([0.9, 0.9]), requires_grad=True)
        ]
        """

        # path rendering
        if num_key_points == 1:
            self.graphic = PointGraphic2d(canvas_shape, **path_kwargs)

        if num_key_points == 2:
            self.graphic = LineGraphic2d(canvas_shape, **path_kwargs)

        if num_key_points >= 3:
            self.graphic = CurveGraphic2d(canvas_shape, **path_kwargs)

        # In the future we'll feed the render to the original
        # classifier to compute our score


    def parameters(self):
        """
        Future: do this properly, something like making the parameters a module?
        """
        model_parameters = super(OpponentModel, self).parameters()
        return list(model_parameters) + self.key_points


    def forward(self):
        output_canvas = self.graphic(self.key_points)

        return output_canvas


    def clamp_endpoints(self):
        with torch.no_grad():
            self.key_points[0].clamp_(0, 1)
            self.key_points[-1].clamp_(0, 1)
