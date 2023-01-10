from typing import Optional, List, Dict, Any

import torch

from modules import StrokeScoreModel
from utils import make_hooked_optimizer, draw_output_and_target

DEVICE = (
    #"mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


def global_search_stroke(
    base_canvas: torch.Tensor,
    initial_inputs: Optional[List[torch.Tensor]],
    score_model: torch.nn.Module,
    target_index: int,
    optimizer_class: torch.optim.Optimizer,
    optimizer_kwargs: Dict[str, Any],
    max_width: float = 10.0,
    min_width: float = 3.5,
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
    best_parameters = list(model.parameters()).copy()
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

        new_width = loss.item() * max_width + min_width
        model.update_graphic_width(new_width)

        print(f"step_num: {step_num}")
        print(f"initial_inputs: {initial_inputs}")
        print(list(model.parameters()))
        print(f"loss: {loss.item()}")

        if save_best and loss.item() < best_loss:
            best_loss = loss.item()
            best_parameters = list(model.parameters()).copy()

        if draw_output:
            draw_output_and_target(canvas_with_graphic[0][0], canvas_with_graphic[0][0])

    if save_best:
        return best_loss, best_parameters

    else:
        return loss.item(), list(model.parameters())
