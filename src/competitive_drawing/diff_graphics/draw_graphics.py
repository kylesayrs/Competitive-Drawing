import cv2
import torch

from competitive_drawing.diff_graphics import StrokeModel
from competitive_drawing.diff_graphics.utils.helpers import draw_output_and_target


def draw_keypoints(base_canvas, keypoints):
    keypoints = keypoints.reshape((1, -1, 2))

    model = StrokeModel(
        base_canvas.shape,
        keypoints,
        max_length=float("Inf"),
        widths=[1.5],
        aa_factors=[1.0],
        num_samples=15,
    )

    output_canvas = model()

    image = draw_output_and_target(base_canvas, output_canvas[0])
    cv2.imshow("output and target", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    canvas_shape = (50, 50)
    base_canvas = cv2.imread("assets/box.png", cv2.IMREAD_GRAYSCALE)
    base_canvas = torch.tensor(base_canvas / 255)

    initial_inputs = torch.tensor([
        [0.6140, 0.3890],
        [0.5048, 0.4737],
        [0.3519, 0.5020],
        [0.5758, 0.4939]
    ])

    draw_keypoints(base_canvas, initial_inputs)
