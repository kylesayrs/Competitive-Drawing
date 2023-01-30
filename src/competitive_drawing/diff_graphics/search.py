from typing import Optional, List, Dict, Any, Tuple

import cv2
import torch
import numpy

from competitive_drawing.diff_graphics import StrokeScoreModel
from .utils.helpers import make_hooked_optimizer, draw_output_and_target

DEVICE = (
    #"mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
DEBUG = False


def grid_search_stroke(
    base_canvas: torch.Tensor,
    grid_shape: Tuple[int, int],
    score_model: torch.nn.Module,
    target_index: int,
    optimizer_class: torch.optim.Optimizer,
    optimizer_kwargs: Dict[str, Any],
    num_keypoints: int = 4,
    max_width: float = 10.0,
    min_width: float = 3.5,
    max_aa: float = 0.35,
    min_aa: float = 0.9,
    max_steps: int = 200,
    return_best: bool = False,
    draw_output: bool = False,
    **model_kwargs,
):
    initial_inputs = torch.from_numpy(numpy.array([
        [
            (numpy.random.random(2) + numpy.array([grid_y, grid_x])) / numpy.array(grid_shape)
            for _ in range(num_keypoints)
        ]
        for grid_y in range(grid_shape[1])
        for grid_x in range(grid_shape[0])
    ], dtype=numpy.float32))


    score, keypoints = search_strokes(
        base_canvas,
        initial_inputs,
        score_model,
        target_index,
        optimizer_class,
        optimizer_kwargs,
        num_keypoints=num_keypoints,
        max_width=max_width,
        min_width=min_width,
        max_aa=max_aa,
        min_aa=min_aa,
        max_steps=max_steps,
        return_best=return_best,
        draw_output=draw_output,
        **model_kwargs,
    )

    if DEBUG:
        print(f"score: {score}")
        print(f"keypoints: {keypoints}")

    return score, keypoints


def search_strokes(
    base_canvas: torch.Tensor,
    initial_inputs: Optional[List[torch.Tensor]],
    score_model: torch.nn.Module,
    target_index: int,
    optimizer_class: torch.optim.Optimizer,
    optimizer_kwargs: Dict[str, Any],
    num_keypoints: int = 4,
    max_width: float = 10.0,
    min_width: float = 3.5,
    max_aa: float = 0.35,
    min_aa: float = 0.9,
    max_steps: int = 200,
    return_best: bool = False,
    draw_output: bool = False,
    **model_kwargs,
):
    initial_inputs = (
        initial_inputs
        if initial_inputs is not None
        else torch.rand(1, num_keypoints, 2)
    )

    model = StrokeScoreModel(
        base_canvas,
        initial_inputs,
        score_model,
        target_index=target_index,
        widths=[max_width for _ in range(initial_inputs.shape[0])],
        aa_factors=[min_aa for _ in range(initial_inputs.shape[0])],
        **model_kwargs
    )
    model = model.to(DEVICE)

    criterion = torch.nn.MSELoss()
    optimizer = make_hooked_optimizer(
        optimizer_class,
        model.constrain_keypoints,
        model.parameters(), **optimizer_kwargs
    )

    return_score = 0.0
    return_keypoints = list(model.parameters()).copy()[0]
    scores = torch.zeros([initial_inputs.shape[0]], dtype=torch.float32, device=DEVICE)
    for step_num in range(max_steps):
        # zero the parameter gradients, set graphics parameters
        optimizer.zero_grad()
        model.update_width_and_anti_aliasing(scores, max_width, min_width, max_aa, min_aa)

        # forward
        canvas_with_graphic, scores = model()

        # backwards, optimize, and constrain (via hook)
        target_score = torch.full([initial_inputs.shape[0]], 1.0, dtype=torch.float32, device=DEVICE)
        loss = criterion(scores, target_score)
        loss.backward()
        optimizer.step()

        best_index = torch.argmax(scores)
        best_score = scores[best_index]
        best_keypoints = list(model.parameters()).copy()[0][best_index]
        if (return_best and best_score > return_score) or not return_best:
            return_score = best_score
            return_keypoints = best_keypoints

        if DEBUG:
            print(f"new_widths: {model.widths}")
            print(f"new_aa_factors: {model.aa_factors}")
            print(f"step_num: {step_num}")
            #print(list(model.parameters())[0])
            #print(f"scores: {scores.tolist()}")
            print(f"best_score: {best_score}")
            print(f"loss: {loss.item()}")

        if draw_output:
            image = draw_output_and_target(base_canvas, torch.sum(canvas_with_graphic, dim=0)[0])
            cv2.imshow("output and target", image)
            cv2.waitKey(0)

    return return_score, return_keypoints
