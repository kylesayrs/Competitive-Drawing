import torch

from modules.CurveGraphic2d import CurveGraphic2d
from utils.BezierCurve import BezierCurve
from utils.helpers import draw_output_and_target


if __name__ == "__main__":
    canvas_shape = (28, 28)

    inputs = [
        torch.tensor([0.0, 0.0]), #
        torch.tensor([0.1, 1.0]),
        torch.tensor([0.3, 0.8]), #
        torch.tensor([1.0, 0.6]),
        torch.tensor([0.8, 0.4]), #
        torch.tensor([0.2, 0.1]),
        torch.tensor([0.5, 0.3]), #
    ]

    graphic = CurveGraphic2d(
        canvas_shape,
        num_samples=15,
        width=3.0,
        anti_aliasing_factor=1.0
    )

    # target
    target_canvas = graphic(inputs)

    # load into curve
    keypoints = [input * torch.tensor(canvas_shape) for input in inputs]
    curve = BezierCurve(keypoints)
    print(f"old arc_length: {curve.arc_length()}")

    # truncate
    curve.truncate(0.8)
    print(f"new arc_length: {curve.arc_length()}")
    new_key_points = curve.key_points

    # forward
    new_inputs = [key_point / torch.tensor(canvas_shape) for key_point in new_key_points]
    truncated_canvas = graphic(new_inputs)

    # draw
    draw_output_and_target(truncated_canvas, target_canvas)
