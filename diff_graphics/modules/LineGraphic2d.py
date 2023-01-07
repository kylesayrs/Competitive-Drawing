from typing import List, Optional, Tuple

import torch

EPSILON = 0.001  # autograd has a hard time with 0.0^x


class LineGraphic2d(torch.nn.Module):
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
            self._get_line_precomputations(key_points)
            if line_precomputations is None
            else line_precomputations
        )

        t_guess = torch.dot(p - p0, p1_minus_p0) / len_line_segment_squared
        t = torch.clamp(t_guess, min=0, max=1)
        projection = p0 + t * p1_minus_p0

        return torch.norm(p - projection)


    def forward(self, key_points):
        # prepare canvas and key_points
        canvas = torch.zeros(self.canvas_shape)
        canvas_shape_tensor = torch.tensor(self.canvas_shape)
        key_points = [key_point * canvas_shape_tensor for key_point in key_points]

        # precompute some values
        line_precomputations = self._get_line_precomputations(key_points)

        # torch doesn't implement a map function, so a single thread will do
        for y in range(0, canvas.shape[0]):
            for x in range(0, canvas.shape[1]):
                p = torch.tensor([y, x], dtype=torch.float32)

                distance = self.get_distance_to_line(
                    p, key_points,
                    line_precomputations=line_precomputations
                )

                if distance < self.width:
                    canvas[y, x] = 1 - (distance / self.max_distance + EPSILON) ** self.anti_aliasing_factor

        return canvas


    def _get_line_precomputations(self, key_points: List[torch.tensor]):
        p1_minus_p0 = key_points[1] - key_points[0]
        len_line_segment_squared = (
            (key_points[1][0] - key_points[0][0]) ** 2 +
            (key_points[1][1] - key_points[0][1]) ** 2
        )

        return p1_minus_p0, len_line_segment_squared
