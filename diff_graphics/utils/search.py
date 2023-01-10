from typing import Optional, List, Dict, Any, Tuple

import torch

from modules.StrokeScoreModel import StrokeScoreModel
from utils.helpers import make_hooked_optimizer, draw_output_and_target

DEVICE = (
    #"mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
DEBUG = True


def grid_search_stroke(
    base_canvas: torch.Tensor,
    grid_shape: Tuple[int, int],
    score_model: torch.nn.Module,
    target_index: int,
    optimizer_class: torch.optim.Optimizer,
    optimizer_kwargs: Dict[str, Any],
    max_width: float = 10.0,
    min_width: float = 3.5,
    max_aa: float = 0.35,
    min_aa: float = 0.9,
    max_steps: int = 200,
    save_best: bool = True,
    draw_output: bool = False,
    **model_kwargs,
):
    best_loss = float("Inf")
    best_keypoints = None
    for grid_x in range(grid_shape[0]):
        for grid_y in range(grid_shape[1]):
            initial_inputs = [
                (torch.rand(2) + torch.tensor([grid_x, grid_y])) / torch.tensor(list(grid_shape))
                for _ in range(4)
            ]

            loss, keypoints = search_stroke(
                base_canvas,
                initial_inputs,
                score_model,
                target_index,
                optimizer_class,
                optimizer_kwargs,
                max_width=max_width,
                min_width=min_width,
                max_aa=max_aa,
                min_aa=min_aa,
                max_steps=max_steps,
                save_best=save_best,
                draw_output=draw_output,
                **model_kwargs,
            )

            if loss < best_loss:
                best_loss = loss
                best_keypoints = keypoints

    if DEBUG:
        print(f"best_loss: {best_loss}")
        print(f"best_keypoints: {best_keypoints}")

    return best_loss, best_keypoints


def search_stroke(
    base_canvas: torch.Tensor,
    initial_inputs: Optional[List[torch.Tensor]],
    score_model: torch.nn.Module,
    target_index: int,
    optimizer_class: torch.optim.Optimizer,
    optimizer_kwargs: Dict[str, Any],
    max_width: float = 10.0,
    min_width: float = 3.5,
    max_aa: float = 0.35,
    min_aa: float = 0.9,
    max_steps: int = 200,
    save_best: bool = True,
    draw_output: bool = False,
    **model_kwargs,
):
    initial_inputs = (
        initial_inputs
        if initial_inputs is not None
        else [torch.rand(2) for _ in range(4)]
    )

    model = StrokeScoreModel(
        base_canvas,
        initial_inputs,
        score_model,
        target_index=target_index,
        width=max_width,
        **model_kwargs
    )
    model = model.to(DEVICE)

    criterion = torch.nn.MSELoss()
    optimizer = make_hooked_optimizer(
        optimizer_class,
        model.constrain_keypoints,
        model.parameters(), **optimizer_kwargs
    )

    best_loss = float("Inf")
    best_keypoints = list(model.parameters()).copy()
    for step_num in range(max_steps):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        canvas_with_graphic, score = model()

        # backwards and optimize
        target_score = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
        loss = criterion(score, target_score)
        loss.backward()
        optimizer.step()

        new_width = loss.item() * max_width + (1.0 - loss.item()) * min_width
        new_aa_factor = loss.item() * max_aa + (1.0 - loss.item()) * min_aa
        model.update_graphic_width(new_width)
        model.update_graph_anti_aliasing_factor(new_aa_factor)

        if DEBUG:
            print(f"new_width: {new_width}")
            print(f"new_aa_factor: {new_aa_factor}")
            print(f"step_num: {step_num}")
            print(f"initial_inputs: {initial_inputs}")
            print(list(model.parameters()))
            print(f"loss: {loss.item()}")

        if save_best and loss.item() < best_loss:
            best_loss = loss.item()
            best_keypoints = list(model.parameters()).copy()

        if draw_output:
            draw_output_and_target(canvas_with_graphic[0][0], canvas_with_graphic[0][0])

    if save_best:
        return best_loss, best_keypoints

    else:
        return loss.item(), list(model.parameters())
