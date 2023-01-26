import cv2
import torch

from competitive_drawing.diff_graphics import CurveGraphic2d
from competitive_drawing.diff_graphics.utils.BezierCurve import BezierCurve
from competitive_drawing.diff_graphics.utils.helpers import draw_output_and_target


if __name__ == "__main__":
    canvas_shape = (28, 28)

    inputs = torch.concat([
        torch.tensor([0.0, 0.0]), #
        torch.tensor([0.1, 1.0]),
        torch.tensor([0.3, 0.8]), #
        torch.tensor([1.0, 0.6]),
        torch.tensor([0.8, 0.4]), #
        torch.tensor([0.2, 0.1]),
        torch.tensor([0.5, 0.3]), #
    ]).reshape((1, -1, 2))

    graphic_layer = CurveGraphic2d(
        canvas_shape,
        num_samples=15,
    )

    # target
    target_canvas = graphic_layer(inputs, widths=[3.0], aa_factors=[1.0])[0]

    # load into curve
    keypoints = [input * torch.tensor(canvas_shape) for input in inputs[0]]
    curve = BezierCurve(keypoints)
    print(f"old arc_length: {curve.arc_length}")

    # truncate
    curve.truncate(0.8)
    print(f"new arc_length: {curve.arc_length}")

    # forward
    new_inputs = torch.concat([
        key_point / torch.tensor(canvas_shape)
        for key_point in curve.key_points
    ]).reshape(1, -1, 2)
    truncated_canvas = graphic_layer(new_inputs, widths=[3.0], aa_factors=[1.0])[0]

    # draw
    image = draw_output_and_target(truncated_canvas, target_canvas)
    cv2.imshow("output and target", image)
    cv2.waitKey(0)
