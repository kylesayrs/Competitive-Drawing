import cv2
import torch

from modules import StrokeScoreModel
from utils import load_score_model, make_hooked_optimizer, draw_output_and_target


if __name__ == "__main__":
    base_canvas = cv2.imread("assets/box.png", cv2.IMREAD_GRAYSCALE)
    base_canvas = torch.tensor(base_canvas / 255)
    """
    initial_inputs = [
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.57, 0.5]),
        torch.tensor([0.55, 0.57]),
        torch.tensor([0.4, 0.6]),
    ]
    """
    initial_inputs = [
        torch.rand(2)
        for _ in range(3)
    ]
    initial_inputs = [
        torch.tensor([0.1791, 0.0679]),
        torch.tensor([0.6804, 0.7775]),
        torch.tensor([0.1237, 0.4318])
    ]
    print(initial_inputs)
    score_model = load_score_model("assets/camera-coffee cup.pth")

    model = StrokeScoreModel(
        base_canvas,
        initial_inputs,
        score_model,
        target_index=0,
        max_length=15.0,
        num_samples=20,
        width=7,#3.5,
        anti_aliasing_factor=0.35
    )

    criterion = torch.nn.MSELoss()
    optimizer = make_hooked_optimizer(
        torch.optim.RMSprop,
        model.constrain_keypoints,
        model.parameters(), lr=0.02
    )

    while True:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        canvas_with_graphic, score = model()

        # backwards and optimize
        target_score = torch.tensor(1.0, dtype=torch.float32)
        loss = criterion(score, target_score)
        loss.backward()
        optimizer.step()

        new_width = loss.item() * 20 + 3.5
        model.update_graphic_width(new_width)

        print(initial_inputs)
        print(list(model.parameters()))
        print(f"loss: {loss.item()}")

        draw_output_and_target(canvas_with_graphic[0][0], canvas_with_graphic[0][0])