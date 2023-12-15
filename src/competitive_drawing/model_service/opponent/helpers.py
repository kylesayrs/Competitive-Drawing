from typing import Callable, Any

import torch
import numpy


def make_hooked_optimizer(
    optimizer_class: torch.optim.Optimizer,
    hook: Callable[[], Any],
    *optimizer_args,
    **optimizer_kwargs
) -> torch.optim.Optimizer:
    """
    Returns an instance of `optimizer_class` which calls `hook` after each step

    :param optimizer_class: class to be instantiated
    :param hook: function to be called after each optimizer step 
    :param optimizer_args: initializer arguments for optimizer
    :param optimizer_kwargs: initializer keyword arguments for optimizer
    :return: instance of `optimizer_class` with included step hook
    """
    class HookedOptimizer(optimizer_class):
        def step(self, closure=None):
            super().step(closure)
            with torch.no_grad():
                hook()

    return HookedOptimizer(*optimizer_args, **optimizer_kwargs)


def draw_output_and_target(
    output_canvas: torch.Tensor,
    target_canvas: torch.Tensor
) -> numpy.ndarray:
    """
    Draws an output and target canvas image. Target areas not covered by the
    output are red, target areas covered by the output are green, and areas
    covered by the output but not the target are blue. Areas covered by neither
    are black.

    :param output_canvas: 2D output canvas
    :param target_canvas: 2D target canvas
    :return: numpy image depicting the output and target canvases
    """
    assert output_canvas.shape == target_canvas.shape
    image = numpy.zeros((*output_canvas.shape, 3))

    output = output_canvas.cpu().detach().numpy()
    target = target_canvas.cpu().detach().numpy()

    image[:, :, 0] = (1.0 - output) * target
    image[:, :, 1] = output * target * 2
    image[:, :, 2] = output * (1.0 - target)

    return image
