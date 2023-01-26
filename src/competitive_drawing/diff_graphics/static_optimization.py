import cv2
import torch

from competitive_drawing.diff_graphics import LineGraphic2d, CurveGraphic2d, StrokeModel
from competitive_drawing.diff_graphics.utils.helpers import make_hooked_optimizer, draw_output_and_target


def make_target_canvas(target: str):
    if target == "sunset":
        target_canvas = torch.zeros(canvas_shape)
        for y in range(canvas_shape[0]):
            target_canvas[y, :] = y / canvas_shape[0]

        return target_canvas

    if target == "line":
        target_points = [
            torch.tensor([0.4, 0.1]),
            torch.tensor([0.9, 0.9]),
        ]
        return LineGraphic2d(canvas_shape, width=4.0)(target_points)

    if target == "curve":
        target_points = torch.concat([
            torch.tensor([0.0, 0.0]), #
            torch.tensor([0.1, 1.0]),
            torch.tensor([0.3, 0.8]), #
            torch.tensor([1.0, 0.6]),
            torch.tensor([0.8, 0.4]), #
            torch.tensor([0.2, 0.1]),
            torch.tensor([0.5, 0.3]), #
        ]).reshape(1, 7, 2)

        graphic_layer = CurveGraphic2d(
            canvas_shape,
            num_samples=30,
        )

        target_canvas = graphic_layer(target_points, widths=[5.0], aa_factors=[0.9])[0]

        return target_canvas


if __name__ == "__main__":
    canvas_shape = (28, 28)
    target = "curve"

    target_canvas = make_target_canvas(target)

    initial_inputs = torch.rand(1, 6, 2)

    model = StrokeModel(
        canvas_shape,
        initial_inputs,
        max_length=50.0,
        num_samples=15,
        widths=[3.0],
        aa_factors=[0.9]
    )

    criterion = torch.nn.MSELoss()
    optimizer = make_hooked_optimizer(
        torch.optim.Adam,
        model.constrain_keypoints,
        model.parameters(), lr=0.1#, momentum=0.85,
    )

    while True:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        output_canvas = model()[0]

        # backwards and optimize
        loss = criterion(output_canvas, target_canvas)
        loss.backward()
        optimizer.step()

        print(list(model.parameters()))
        print(f"loss: {loss.item()}")
        image = draw_output_and_target(output_canvas, target_canvas)
        cv2.imshow("output and target", image)
        cv2.waitKey(0)
