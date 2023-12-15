from typing import Optional, List, Dict, Any, Tuple

import cv2
import torch
import numpy

from .models.StrokeScoreModel import StrokeScoreModel
from .helpers import make_hooked_optimizer
from .SearchParameters import SearchParameters


def grid_search_stroke(
    base_canvas: torch.Tensor,
    score_model: torch.nn.Module,
    target_index: int,
    optimizer_class: torch.optim.Optimizer,
    optimizer_kwargs: Dict[str, Any],
    search_parameters: SearchParameters,
    **model_kwargs,
) -> Tuple[float, torch.Tensor]:
    """
    Search for an optimal stroke by randomly initializing strokes within a grid
    pattern. Score is optimized with respect to the `target_index` of the score
    model

    :param base_canvas: canvas upon which strokes are drawn
    :param grid_shape: number of rows and columns of grid
    :param score_model: model whose output is used as an objective function
    :param target_index: index of `score_model` output to use as objective function
    :param optimizer_class: class used to optimize stroke
    :param optimizer_kwargs: arguments used to initialize optimizer
    :param search_parameters: parameters used to search for curves
    :return: best score and curve keypoints
    """
    # randomly initialize keypoints on grid
    initial_inputs = _initialize_grid(search_parameters)

    # create model with initial keypoints
    model = StrokeScoreModel(
        base_canvas,
        initial_inputs,
        score_model,
        target_index=target_index,
        widths=[search_parameters.max_width for _ in range(initial_inputs.shape[0])],
        aa_factors=[search_parameters.min_aa for _ in range(initial_inputs.shape[0])],
        **model_kwargs
    )
    model = model.to(search_parameters.device)

    # create optimizer w.r.t. MSE
    criterion = torch.nn.MSELoss()
    optimizer = make_hooked_optimizer(
        optimizer_class,
        model.constrain_keypoints,
        model.parameters(), **optimizer_kwargs
    )

    # optimize
    return_score = 0.0
    return_keypoints = list(model.parameters()).copy()[0]
    scores = torch.zeros([initial_inputs.shape[0]], dtype=torch.float32, device=search_parameters.device)
    for _step_num in range(search_parameters.max_steps):
        # zero the parameter gradients, set graphics parameters
        optimizer.zero_grad()
        model.update_width_and_anti_aliasing(
            scores,
            search_parameters.max_width,
            search_parameters.min_width,
            search_parameters.max_aa,
            search_parameters.min_aa
        )

        # forward
        _canvas_with_graphic, scores = model()

        # backwards, optimize, and constrain (via hook)
        target_score = torch.full([initial_inputs.shape[0]], 1.0, dtype=torch.float32, device=search_parameters.device)
        loss = criterion(scores, target_score)
        loss.backward()
        optimizer.step()

        best_index = torch.argmax(scores)
        best_score = scores[best_index]
        best_keypoints = list(model.parameters()).copy()[0][best_index]
        if (
            (search_parameters.return_best and best_score > return_score) or
            not search_parameters.return_best
        ):
            return_score = best_score
            return_keypoints = best_keypoints

    return return_score, return_keypoints



def _initialize_grid(search_parameters: SearchParameters) -> torch.Tensor:
    """
    Initilize keypoints on grid. Each set of keypoints is confined to one block
    within the grid

    :param search_parameters: parameters which define search
    :return: keypoints randomly initialized in grid
    """
    grid_shape = numpy.array(search_parameters.grid_shape)

    return torch.from_numpy(numpy.array([
        [
            (numpy.random.random(2) + numpy.array([grid_y, grid_x])) / grid_shape
            for _ in range(search_parameters.num_keypoints)
        ]
        for grid_y in range(grid_shape[1])
        for grid_x in range(grid_shape[0])
    ], dtype=numpy.float32))
