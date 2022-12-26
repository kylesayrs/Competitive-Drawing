import cv2
import torch
import numpy

from OpponentModel import OpponentModel
from LineGraphic2d import LineGraphic2d
from CurveGraphic2d import CurveGraphic2d


def draw_output_and_target(output_canvas, target_canvas):
    assert output_canvas.shape == target_canvas.shape
    image = numpy.zeros((*output_canvas.shape, 3))

    output = output_canvas.detach().numpy()
    target = target_canvas.detach().numpy()

    image[:, :, 0] = cv2.bitwise_and(1.0 - output, target)
    image[:, :, 1] = cv2.bitwise_and(output, target)
    image[:, :, 2] = cv2.bitwise_and(output, 1.0 - target)

    cv2.imshow("output and target", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    canvas_shape = (28, 28)

    """ sunset target
    target_canvas = torch.zeros(canvas_shape)
    for y in range(canvas_shape[0]):
        target_canvas[y, :] = y / canvas_shape[0]
    """

    """ exact line target
    target_points = [
        torch.tensor([0.4, 0.1]),
        torch.tensor([0.9, 0.9]),
    ]
    target_canvas = LineGraphic2d(canvas_shape, width=4.0)(target_points)
    """

    #""" exact line curve
    target_points = [
        torch.tensor([0.0, 0.0]), #
        torch.tensor([0.1, 1.0]),
        torch.tensor([0.3, 0.8]), #
        torch.tensor([1.0, 0.6]),
        torch.tensor([0.8, 0.4]), #
        torch.tensor([0.2, 0.1]),
        torch.tensor([0.5, 0.3]), #
    ]
    target_canvas = CurveGraphic2d(
        canvas_shape,
        num_samples=30,
        width=5.0,
        anti_aliasing_factor=0.25,
    )(target_points)
    #"""

    model = OpponentModel(
        canvas_shape=canvas_shape,
        num_key_points=7,
        num_samples=15,
        width=3.0,
        anti_aliasing_factor=0.25
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8, momentum=0.9)

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
