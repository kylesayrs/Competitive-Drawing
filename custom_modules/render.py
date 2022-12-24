import cv2
import torch

def render_line(canvas, p0, p1):
    """
    Renders a line from p0 to p1 on the canvas.
    The result is differentiable for all points
    with respect to the inputs

    len_of_projection = torch.dot(p - p0, p1 - p0) / torch.norm(p1 - p0)
    t = len_of_projection / torch.norm(p1 - p0)  |  0 ≤ t ≤ 1
    projection = p0 + t * (p1 - p0)
    distance_to_line = torch.norm(p - projection)
    """

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

            canvas[x, y] += torch.norm(p - projection)

    return canvas


if __name__ == "__main__":
    canvas = torch.zeros((28, 28))
    p0 = torch.tensor([0.0, 0.0])
    p1 = torch.tensor([14.0, 1.0])

    canvas = render_line(canvas, p0, p1)

    cv2.imshow("canvas", canvas.numpy())
    cv2.waitKey(0)
