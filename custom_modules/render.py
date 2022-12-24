import cv2
import torch


class LineRaster2d(torch.nn.Module):
    """
    Renders a line from p0 to p1 on the canvas.
    The result is differentiable at all points
    with respect to the inputs (although we restrict
    rendering based on line_width)

    len_of_projection = torch.dot(p - p0, p1 - p0) / torch.norm(p1 - p0)
    t = len_of_projection / torch.norm(p1 - p0)  |  0 ≤ t ≤ 1
    projection = p0 + t * (p1 - p0)
    distance_to_line = torch.norm(p - projection)
    """

    def __init__(self, canvas_shape, line_width=1.0):
        super().__init__()
        self.canvas_shape = canvas_shape
        self.line_width = line_width

    def forward(self, p0, p1):
        p0 = p0 * torch.tensor(self.canvas_shape)
        p1 = p1 * torch.tensor(self.canvas_shape)
        canvas = torch.zeros(self.canvas_shape)

        # precompute some values
        p1_minus_p0 = p1 - p0
        len_line_segment_squared = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2

        # torch doesn't implement a map function, so a single thread will do
        for y in range(0, canvas.shape[0]):
            for x in range(0, canvas.shape[1]):
                p = torch.tensor([x, y], dtype=torch.float32)

                p_minus_p0 = p - p0
                t_guess = torch.dot(p_minus_p0, p1_minus_p0) / len_line_segment_squared
                t = torch.clamp(t_guess, min=0, max=1)
                projection = p0 + t * p1_minus_p0

                distance = torch.norm(p - projection)
                if distance < self.line_width:
                    canvas[x, y] = self.line_width - distance

        return canvas


class OpponentModel(torch.nn.Module):
    def __init__(self, canvas_shape, line_width=2.0):
        super(OpponentModel, self).__init__()

        # In the future these will be computed outputs from a
        # model that takes the original canvas as input
        self.p0 = torch.nn.Parameter(torch.rand(2), requires_grad=True)
        self.p1 = torch.nn.Parameter(torch.rand(2), requires_grad=True)

        # line rendering
        self.line_raster = LineRaster2d(canvas_shape, line_width=2.0)

        # In the future we'll feed the render to the original
        # classifier to compute our score

    def forward(self):
        p0 = torch.sigmoid(self.p0 * 1)
        p1 = torch.sigmoid(self.p1 * 1)
        output_canvas = self.line_raster(p0, p1)

        return output_canvas


# Notes: There are many contraints to take into account
# 1. endpoints should be between 0 and 1 (or 0 and 28)
#    a. model.p0.data.clamp_(0.01, 0.99)
#       model.p1.data.clamp_(0.01, 0.99)
#       although you lose the gradient if it goes off
#    b. alternatively, the function is differentiable everywhere, although
#       frankly I don't want rendering to affect every pixel, even during training
#       Note: I tried this, and sometimes the gradient is so weak it doesn't affect loss
#       I don't think this is enough for it to recover
#    c. I just don't want endpoints to end up off canvas because then
#       it messes with length calculations and if the line goes out of bounds
#       then there is no gradient
#    I think a happy medium is applying a sigmoid potentially with some alpha < 1
#    This clamps it and still provides some gradient
# 2. endpoints must have a length <= some value
# 3. rendered outputs must be between 0 and 1
#
# worst case it goes out of bounds and the AI doesn't use its entire distance


if __name__ == "__main__":
    canvas_shape = (28, 28)

    """ sunset target
    target_canvas = torch.zeros(canvas_shape)
    for y in range(canvas_shape[0]):
        target_canvas[y, :] = y / canvas_shape[0]
    """

    """ exact line target """
    target_p0 = torch.tensor([0.1, 0.1])
    target_p1 = torch.tensor([0.9, 0.9])
    target_canvas = LineRaster2d(canvas_shape, line_width=4.0)(target_p0, target_p1)

    cv2.imshow("target_canvas", target_canvas.numpy())
    cv2.waitKey(0)

    model = OpponentModel(canvas_shape, line_width=2.0)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=3.0)

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
        cv2.imshow("output_canvas", output_canvas.detach().numpy())
        cv2.waitKey(0)
