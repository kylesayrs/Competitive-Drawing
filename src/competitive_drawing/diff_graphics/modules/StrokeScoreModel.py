from typing import List, Tuple

import torch

from .StrokeModel import StrokeModel
from competitive_drawing import Settings


class StrokeScoreModel(StrokeModel):
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
        base_canvas: torch.Tensor,
        initial_inputs: List[torch.Tensor],
        score_model: torch.nn.Module,
        target_index: int,
        max_length: float,
        **path_kwargs,
    ):

        super().__init__(
            canvas_shape=base_canvas.shape,
            initial_inputs=initial_inputs,
            max_length=max_length,
            **path_kwargs,
        )

        self.base_canvas = base_canvas.to(torch.float32)
        self.score_model = score_model.float()
        self.score_model.eval()
        self.target_index = target_index

        # freeze score model
        for param in self.score_model.parameters():
            param.requires_grad = False


    def forward(self):
        graphic = super().forward()
        canvas_with_graphic = self.base_canvas + graphic
        canvas_with_graphic = torch.reshape(canvas_with_graphic.to(torch.float32), (-1, 1, *self.base_canvas.shape))
        logits, scores = self.score_model(canvas_with_graphic)

        target_scores = scores[:, self.target_index]

        return canvas_with_graphic, target_scores


    def to(self, device):
        super().to(device)
        self.base_canvas = self.base_canvas.to(device)
        self.score_model = self.score_model.to(device)

        return self
