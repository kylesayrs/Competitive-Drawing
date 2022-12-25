from typing import List, Optional

import cv2
import torch


class PathRaster2d(torch.nn.Module):
    """
    """

    def __init__(
        self,
        canvas_shape: List[int],
        line_width: float = 1.0,
        anti_aliasing: str = "linear",
        num_path_samples: int = 10,
        path_sample_method: str = "uniform"
    ):
        super().__init__()
        self.canvas_shape = canvas_shape
        self.line_width = line_width
        self.anti_aliasing = anti_aliasing
        self.num_path_samples = num_path_samples
        self.path_sample_method = path_sample_method

        self.max_distance = torch.norm(
            torch.tensor(canvas_shape) - torch.tensor([0.0, 0.0])
        )

        if self.anti_aliasing not in ["linear", "quadratic", "global"]:
            raise ValueError("anti_aliasing must be 'linear', 'quadratic' or 'global'")

        if self.path_sample_method not in ["uniform", "stochastic"]:
            raise ValueError("path_sample_method must be 'uniform' or 'stochastic'")


    def get_distance_to_point(self, p: torch.tensor, key_points: List[torch.tensor]):
        """
        Obvious closed form
        """
        return torch.norm(p - key_points[0])


    def get_distance_to_line(
        self,
        p: torch.tensor,
        key_points: List[torch.tensor],
        p1_minus_p0: Optional[float] = None,
        len_line_segment_squared: Optional[float] = None
    ):
        """
        This one has a closed form solution
        """
        p0 = key_points[0]
        p1 = key_points[1]
        p1_minus_p0 = p1 - p0 if p1_minus_p0 is None else p1_minus_p0
        len_line_segment_squared = (
            (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
            if len_line_segment_squared is None
            else len_line_segment_squared
        )

        t_guess = torch.dot(p - p0, p1_minus_p0) / len_line_segment_squared
        t = torch.clamp(t_guess, min=0, max=1)
        projection = p0 + t * p1_minus_p0

        return torch.norm(p - projection)


    def sample_distance_to_path(
        self,
        p: torch.tensor,
        key_points: List[torch.tensor]
    ):
        """
        """
        assert len(key_points) > 0

        #path_sample_method



    def forward(self, key_points):
        # prepare canvas and key_points
        canvas = torch.zeros(self.canvas_shape)
        canvas_shape_tensor = torch.tensor(self.canvas_shape)
        key_points = [key_point * canvas_shape_tensor for key_point in key_points]

        # precompute some values if applicable
        if len(key_points) == 2:
            p1_minus_p0 = key_points[1] - key_points[0]
            len_line_segment_squared = (
                (key_points[1][0] - key_points[0][0]) ** 2 +
                (key_points[1][1] - key_points[0][1]) ** 2
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
                        p1_minus_p0=p1_minus_p0,
                        len_line_segment_squared=len_line_segment_squared
                    )
                else:
                    distance = self.sample_distance_to_path(p, *key_points)

                if self.anti_aliasing == "linear" and distance < self.line_width:
                    canvas[y, x] = self.line_width - distance

                if self.anti_aliasing == "quadratic":
                    # TODO: move 1/3 to an argument
                    canvas[y, x] = 1 - (distance / self.max_distance) ** (1 / 3)

                if self.anti_aliasing == "global":
                    canvas[y, x] = 1 - (distance / self.max_distance)

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

    def __init__(self, canvas_shape, line_width=2.0):
        super(OpponentModel, self).__init__()

        # key points to optimize
        self.key_points = [
            torch.nn.Parameter(torch.rand(2), requires_grad=True)
            for _ in range(1)
        ]

        # line rendering
        self.line_raster = PathRaster2d(canvas_shape, line_width=2.0, anti_aliasing="global")

        # In the future we'll feed the render to the original
        # classifier to compute our score

    def parameters(self):
        """
        Future: do this properly, something like making the parameters a module?
        """
        model_parameters = super(OpponentModel, self).parameters()
        return list(model_parameters) + self.key_points

    def forward(self):
        key_points = [
            torch.sigmoid(key_point * 1)
            for key_point in self.key_points
        ]
        print(key_points)

        output_canvas = self.line_raster(self.key_points)

        return output_canvas


if __name__ == "__main__":
    canvas_shape = (28, 28)

    #""" sunset target
    target_canvas = torch.zeros(canvas_shape)
    for y in range(canvas_shape[0]):
        target_canvas[y, :] = y / canvas_shape[0]
    #"""

    """ exact line target
    target_p0 = torch.tensor([0.4, 0.1])
    target_p1 = torch.tensor([0.9, 0.9])
    target_canvas = PathRaster2d(canvas_shape, line_width=4.0)([target_p0, target_p1])
    """

    cv2.imshow("target_canvas", target_canvas.numpy())
    cv2.waitKey(0)

    model = OpponentModel(canvas_shape, line_width=2.0)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # 1.0 for global, 0.1 for linear

    while True:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        output_canvas = model()

        # backwards and optimize
        loss = criterion(output_canvas, target_canvas)
        loss.backward()
        optimizer.step()

        #print(list(model.parameters()))
        print(f"loss: {loss.item()}")
        cv2.imshow("output_canvas", output_canvas.detach().numpy())
        cv2.waitKey(0)
