import cv2
import torch

from modules.StrokeModel import StrokeModel
from utils.helpers import draw_output_and_target


if __name__ == "__main__":
    canvas_shape = (50, 50)
    base_canvas = cv2.imread("assets/box.png", cv2.IMREAD_GRAYSCALE)
    base_canvas = torch.tensor(base_canvas / 255)

    initial_inputs = [
        torch.tensor([0.6140, 0.3890]),
        torch.tensor([0.5048, 0.4737]),
        torch.tensor([0.3519, 0.5020]),
        torch.tensor([0.5758, 0.4939])
    ]

    model = StrokeModel(
        canvas_shape,
        initial_inputs,
        max_length=float("Inf"),
        num_samples=15,
        width=1.5,
        anti_aliasing_factor=1.0
    )

    output_canvas = model()

    draw_output_and_target(base_canvas, output_canvas)
