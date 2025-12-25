import torch
import random
import cairocffi as cairo


class VecToRaster:
    def __init__(
        self,
        side: int,
        line_diameter: float,
        padding: float,
        min_scale: float,
        max_scale: float,
    ):
        self.original_side = 256
        self.surface = cairo.ImageSurface(cairo.FORMAT_A8, side, side)
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_antialias(cairo.ANTIALIAS_BEST)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.set_line_width(self.original_side * line_diameter)

        # scale to match the new size
        # add padding at the edges for the line_diameter
        # and add additional padding to account for antialiasing
        total_padding = self.original_side * (padding * 2.0 + line_diameter)
        new_scale = side / (self.original_side + total_padding)
        self.ctx.scale(new_scale, new_scale)
        self.ctx.translate(total_padding / 2.0, total_padding / 2.0)
        assert self.surface.get_height() == self.surface.get_width() == side

        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, vector_image: list[list[list[int]]]) -> torch.Tensor:
        # vector_image: list of strokes
        # stroke: xs, ys
        x_max = max([max(stroke[0]) for stroke in vector_image])
        y_max = max([max(stroke[1]) for stroke in vector_image])

        # scale points
        scale = rand(self.min_scale, self.max_scale)
        x_max *= scale
        y_max *= scale

        # random padding
        x_pad = rand(0, self.original_side - x_max)
        y_pad = rand(0, self.original_side - y_max)   

        # apply scale and padding
        vector_image = [
            [
                [x * scale + x_pad for x in xs],
                [y * scale + y_pad for y in ys]
            ]
            for xs, ys in vector_image
        ]

        # clear background
        self.ctx.set_source_rgba(0, 0, 0, 0)
        self.ctx.paint()

        # draw strokes, this is the most cpu-intensive part
        self.ctx.set_source_rgba(0, 0, 0, 1)
        for xv, yv in vector_image:
            self.ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                self.ctx.line_to(x, y)
            self.ctx.stroke()

        buffer = self.surface.get_data()
        data = torch.asarray(buffer, dtype=torch.uint8)
        image = data.reshape(self.surface.get_height(), self.surface.get_width())
        return image

def rand(a, b):
    return (b - a) * random.random() + a