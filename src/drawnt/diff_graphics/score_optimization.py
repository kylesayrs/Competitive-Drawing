import cv2
import torch

from drawnt.diff_graphics.utils.load_score_model import load_score_model
from drawnt.diff_graphics.utils.search import search_stroke, grid_search_stroke


if __name__ == "__main__":
    mode = "grid"

    base_canvas = cv2.imread("assets/box.png", cv2.IMREAD_GRAYSCALE)
    base_canvas = torch.tensor(base_canvas / 255)

    score_model = load_score_model("assets/camera-coffee cup.pth")

    optimizer_kwargs = {
        "lr": 0.02
    }

    target_index = 0
    max_length = 15.0
    if mode == "global":
        loss, keypoints = search_stroke(
            base_canvas,
            None,
            score_model,
            target_index,
            torch.optim.RMSprop,
            optimizer_kwargs,
            max_width=10.0,
            min_width=1.5,
            max_aa=0.35,
            min_aa=0.9,
            max_steps=500,
            save_best=True,
            draw_output=True,
            max_length=max_length,
        )

    elif mode == "grid":
        loss, keypoints = grid_search_stroke(
            base_canvas,
            (3, 3),
            score_model,
            target_index,
            torch.optim.Adamax,
            optimizer_kwargs,
            max_width=5.0,
            min_width=1.5,
            max_aa=0.35,
            min_aa=0.9,
            max_steps=100,
            save_best=True,
            draw_output=True,
            max_length=max_length,
        )

    else:
        raise ValueError(f"Unknown mode {mode}")

    print(f"loss: {loss}")
    print(f"keypoints: {keypoints}")
