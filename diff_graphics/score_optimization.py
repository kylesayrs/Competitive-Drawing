import cv2
import torch

from utils import load_score_model, global_search_stroke


if __name__ == "__main__":
    base_canvas = cv2.imread("assets/box.png", cv2.IMREAD_GRAYSCALE)
    base_canvas = torch.tensor(base_canvas / 255)

    score_model = load_score_model("assets/camera-coffee cup.pth")

    optimizer_kwargs = {
        "lr": 0.02
    }

    initial_inputs = None
    target_index = 0
    max_width = 15.0
    global_search_stroke(
        base_canvas,
        initial_inputs,
        score_model,
        target_index,
        torch.optim.RMSprop,
        optimizer_kwargs,
        max_width=10.0,
        min_width=3.5,
        max_steps=200,
        save_best=True,
        draw_output=True,
        max_length=15.0,
    )
