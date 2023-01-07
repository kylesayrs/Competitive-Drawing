from typing import List

import torch

EPSILON = 0.001  # autograd has a hard time with 0.0^x


class PointGraphic2d(torch.nn.Module):
    """

    """

    def __init__(
        self,
        canvas_shape: List[int],
        width: float = 1.0,
        anti_aliasing_factor: str = 1.0,
    ):
        super().__init__()
        self.canvas_shape = canvas_shape
        self.width = width
        self.anti_aliasing_factor = anti_aliasing_factor

        self.max_distance = torch.norm(
            torch.tensor(canvas_shape) - torch.tensor([0.0, 0.0])
        )


    def get_distance_to_point(self, p: torch.tensor, key_points: List[torch.tensor]):
        """
        Obvious closed form
        """
        return torch.norm(p - key_points[0])


    def forward(self, key_points):
        # prepare canvas and key_points
        canvas = torch.zeros(self.canvas_shape)
        canvas_shape_tensor = torch.tensor(self.canvas_shape)
        key_points = [key_point * canvas_shape_tensor for key_point in key_points]

        # torch doesn't implement a map function, so a single thread will do
        for y in range(0, canvas.shape[0]):
            for x in range(0, canvas.shape[1]):
                p = torch.tensor([y, x], dtype=torch.float32)

                distance = self.get_distance_to_point(p, key_points)

                if distance < self.width:
                    canvas[y, x] = 1 - (distance / self.max_distance + EPSILON) ** self.anti_aliasing_factor

        return canvas
