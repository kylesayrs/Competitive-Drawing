import cv2
import torch
import numpy

from OpponentModel import OpponentModel
from CurveGraphic2d import CurveGraphic2d


def draw_output_and_target(output_canvas, target_canvas):
    assert output_canvas.shape == target_canvas.shape
    image = numpy.zeros((*output_canvas.shape, 3))

    image[:, :, 0] = output_canvas.detach().numpy()
    image[:, :, 1] = output_canvas.detach().numpy()
    image[:, :, 2] = target_canvas.detach().numpy()

    cv2.imshow("output and target", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    canvas_shape = (28, 28)

    """ sunset target
    target_canvas = torch.zeros(canvas_shape)
    for y in range(canvas_shape[0]):
        target_canvas[y, :] = y / canvas_shape[0]
    """

    #""" exact line target
    target_p0 = torch.tensor([0.4, 0.1])
    target_p1 = torch.tensor([0.9, 0.9])
    target_canvas = CurveGraphic2d(canvas_shape, width=4.0)([target_p0, target_p1])
    #"""

    model = OpponentModel(
        canvas_shape=canvas_shape,
        num_key_points=4,
        width=3.0,
        anti_aliasing_factor=0.25
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

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
