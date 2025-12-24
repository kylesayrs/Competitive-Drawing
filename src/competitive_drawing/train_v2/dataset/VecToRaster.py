import torch
import cairocffi as cairo


class VecToRaster:
    def __init__(
        self,
        side: int = 28,
        line_diameter: int = 16,
        padding: int = 16,
        bg_color=(0,0,0),
        fg_color=(1,1,1)
    ):
        self.original_side = 256
        self.bg_color = bg_color
        self.fg_color = fg_color

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_antialias(cairo.ANTIALIAS_BEST)
        self.ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        self.ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self.ctx.set_line_width(line_diameter)

        # scale to match the new size
        # add padding at the edges for the line_diameter
        # and add additional padding to account for antialiasing
        total_padding = padding * 2.0 + line_diameter
        new_scale = side / (self.original_side + total_padding)
        self.ctx.scale(new_scale, new_scale)
        self.ctx.translate(total_padding / 2.0, total_padding / 2.0)
        assert self.surface.get_height() == self.surface.get_width() == side

    def __call__(self, vector_image: list[list[list[int]]]) -> torch.Tensor:
        # vector_image: list of strokes
        # stroke: xs, ys

        # clear background
        self.ctx.set_source_rgb(*self.bg_color)
        self.ctx.paint()

        # draw strokes, this is the most cpu-intensive part
        self.ctx.set_source_rgb(*self.fg_color)
        for xv, yv in vector_image:
            self.ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                self.ctx.line_to(x, y)
            self.ctx.stroke()

        buffer = self.surface.get_data()
        data = torch.asarray(buffer, dtype=torch.uint8)[::4]
        image = data.reshape(self.surface.get_height(), self.surface.get_width())
        return image
