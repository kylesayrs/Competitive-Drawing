import cv2
import torch


class LineRaster2d(torch.nn.Module):
    """
    Renders a line from p0 to p1 on the canvas.
    The result is differentiable for all points
    with respect to the inputs

    len_of_projection = torch.dot(p - p0, p1 - p0) / torch.norm(p1 - p0)
    t = len_of_projection / torch.norm(p1 - p0)  |  0 ≤ t ≤ 1
    projection = p0 + t * (p1 - p0)
    distance_to_line = torch.norm(p - projection)
    """

    def __init__(self, canvas_shape, line_width=1.0):
        super().__init__()
        self.canvas_shape = canvas_shape
        self.line_width = torch.nn.Parameter(torch.tensor(line_width))

    def forward(self, p0, p1):
        #p0 = torch.clamp(p0, min=0, max=1) * torch.tensor(self.canvas_shape)
        #p1 = torch.clamp(p1, min=0, max=1) * torch.tensor(self.canvas_shape)
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


# Notes: There are many contraints to take into account
# 1. endpoints should be between 0 and 1 (or 0 and 28)
#    a. alternatively, the function is differentiable everywhere
# 2. endpoints must have a length <= some value
# 3. rendered outputs must be between 0 and 1
#
# I should swap the line function to a non-differentiable-everywhere version,
# since this is most likely going to be how I implement the curve and frankly
# I don't want rendering to affect every pixel, even during training


if __name__ == "__main__":
    canvas_shape = (28, 28)
    p0 = torch.nn.Parameter(torch.tensor([0.01, 0.01]), requires_grad=True)
    p1 = torch.nn.Parameter(torch.tensor([0.7, 0.2]), requires_grad=True)

    target_canvas = torch.zeros(canvas_shape)
    for y in range(canvas_shape[0]):
        target_canvas[y, :] = y / canvas_shape[0]

    #cv2.imshow("target_canvas", target_canvas.numpy())
    #cv2.waitKey(0)

    module = LineRaster2d(canvas_shape, line_width=2.0)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(list(module.parameters()) + [p0, p1], lr=0.1)

    while True:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        output_canvas = module(p0, p1)

        # backwards + optimize
        loss = criterion(output_canvas, target_canvas)
        loss.backward()
        optimizer.step()

        print(f"p0: {p0} p1: {p1}")
        print(f"loss: {loss.item()}")
        cv2.imshow("output_canvas", output_canvas.detach().numpy())
        cv2.waitKey(0)
