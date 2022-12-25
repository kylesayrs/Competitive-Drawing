from typing import List, Optional, Tuple

import cv2
import torch
import numpy

from helpers import (
    get_line_precomputations,
    get_bezier_curve_precomputations
)


class PathRaster2d(torch.nn.Module):
    """
    """

    def __init__(
        self,
        canvas_shape: List[int],
        line_width: float = 1.0,
        anti_aliasing_factor: str = 1.0,
        num_path_samples: int = 8,
        path_sample_method: str = "uniform"
    ):
        super().__init__()
        self.canvas_shape = canvas_shape
        self.line_width = line_width
        self.anti_aliasing_factor = anti_aliasing_factor
        self.num_path_samples = num_path_samples
        self.path_sample_method = path_sample_method

        self.max_distance = torch.norm(
            torch.tensor(canvas_shape) - torch.tensor([0.0, 0.0])
        )

        if self.num_path_samples < 3:
            raise ValueError("num_path_samples must be greater than 3")

        if self.path_sample_method not in ["uniform", "uniform_t", "stochastic"]:
            raise ValueError(
                "path_sample_method must be 'uniform', 'uniform_t', or 'stochastic'"
            )


    def get_distance_to_point(self, p: torch.tensor, key_points: List[torch.tensor]):
        """
        Obvious closed form
        """
        return torch.norm(p - key_points[0])


    def get_distance_to_line(
        self,
        p: torch.tensor,
        key_points: List[torch.tensor],
        line_precomputations: Optional[Tuple[float, float]] = None,
    ):
        """
        This one has a closed form solution
        """
        p0 = key_points[0]
        p1 = key_points[1]
        p1_minus_p0, len_line_segment_squared = (
            get_line_precomputations(key_points)
            if line_precomputations is None
            else line_precomputations
        )

        t_guess = torch.dot(p - p0, p1_minus_p0) / len_line_segment_squared
        t = torch.clamp(t_guess, min=0, max=1)
        projection = p0 + t * p1_minus_p0

        return torch.norm(p - projection)


    def sample_distance_to_path(
        self,
        p: torch.tensor,
        key_points: List[torch.tensor],
        b_curve_precomputations: Optional[List[torch.tensor]] = None,
    ):
        """
        """
        assert len(key_points) > 0

        sample_points = (
            get_bezier_curve_precomputations(key_points, self.num_path_samples)
            if b_curve_precomputations is None
            else b_curve_precomputations
        )

        return min([torch.norm(p - sample_point) for sample_point in sample_points])


    def forward(self, key_points):
        # prepare canvas and key_points
        canvas = torch.zeros(self.canvas_shape)
        canvas_shape_tensor = torch.tensor(self.canvas_shape)
        key_points = [key_point * canvas_shape_tensor for key_point in key_points]

        # precompute some values if applicable
        if len(key_points) == 2:
            line_precomputations = get_line_precomputations(key_points)

        if len(key_points) >= 3:
            b_curve_precomputations = get_bezier_curve_precomputations(
                key_points,
                self.num_path_samples,
                self.path_sample_method
            )

        # torch doesn't implement a map function, so a single thread will do
        for y in range(0, canvas.shape[0]):
            for x in range(0, canvas.shape[1]):
                p = torch.tensor([y, x], dtype=torch.float32)

                if len(key_points) == 1:
                    distance = self.get_distance_to_point(p, key_points)

                elif len(key_points) == 2:
                    distance = self.get_distance_to_line(
                        p, key_points,
                        line_precomputations=line_precomputations
                    )
                else:
                    distance = self.sample_distance_to_path(
                        p, key_points,
                        b_curve_precomputations=b_curve_precomputations
                    )

                if distance < self.line_width:
                    canvas[y, x] = 1 - (distance / self.max_distance) ** self.anti_aliasing_factor

        return canvas


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
        line_width: float = 2.0,
        anti_aliasing_factor: float = 1.0
    ):
        super(OpponentModel, self).__init__()

        # key points to optimize
        self.key_points = [
            torch.nn.Parameter(torch.rand(2), requires_grad=True)
            for _ in range(4)
        ]

        # path rendering
        self.path_raster = PathRaster2d(
            canvas_shape,
            line_width=line_width,
            anti_aliasing_factor=anti_aliasing_factor,
        )

        # In the future we'll feed the render to the original
        # classifier to compute our score

    def parameters(self):
        """
        Future: do this properly, something like making the parameters a module?
        """
        model_parameters = super(OpponentModel, self).parameters()
        return list(model_parameters) + self.key_points

    def forward(self):
        # clamp endpoints
        key_points =[
            torch.clamp(key_point, min=0, max=1)
            if key_point_i == 0 or key_point_i == len(self.key_points) - 1
            else key_point
            for key_point_i, key_point in enumerate(self.key_points)
        ]

        output_canvas = self.path_raster(key_points)

        return output_canvas


def draw_output_and_target(output_canvas, target_canvas):
    assert output_canvas.shape == target_canvas.shape
    image = numpy.zeros((*output_canvas.shape, 3))

    image[:, :, 0] = output_canvas.detach().numpy()
    image[:, :, 1] = target_canvas.detach().numpy()

    cv2.imshow("output and target", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    canvas_shape = (28, 28)

    """ sunset target
    target_canvas = torch.zeros(canvas_shape)
    for y in range(canvas_shape[0]):
        target_canvas[y, :] = y / canvas_shape[0]
    """

    #""" exact line target
    target_p0 = torch.tensor([0.4, 0.1])
    target_p1 = torch.tensor([0.9, 0.9])
    target_canvas = PathRaster2d(canvas_shape, line_width=4.0)([target_p0, target_p1])
    #"""

    cv2.imshow("target_canvas", target_canvas.numpy())
    cv2.waitKey(0)

    model = OpponentModel(canvas_shape, line_width=4.0, anti_aliasing_factor=0.25)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    while True:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        output_canvas = model()

        # backwards and optimize
        loss = criterion(output_canvas, target_canvas)
        loss.backward()
        optimizer.step()

        print(list(model.parameters()))
        print(f"loss: {loss.item()}")
        draw_output_and_target(output_canvas, target_canvas)
