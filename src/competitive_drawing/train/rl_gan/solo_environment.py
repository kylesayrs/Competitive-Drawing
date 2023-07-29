from typing import Optional, Dict

import torch

from competitive_drawing.train.rl_gan import EnvironmentConfig
from competitive_drawing.diff_graphics import CurveGraphic2d


class SoloEnvironment():
    def __init__(
        self,
        environment_config: EnvironmentConfig,
        base_image: Optional[torch.tensor] = None,
        steps_left: Optional[int] = None,
    ):
        if base_image is None != steps_left is None:
            raise ValueError("Provide both base_image and steps_left")

        super().__init__()
        self.config = environment_config
        if base_image is not None:
            self.base_image = base_image.to(self.config.device)
        else:
            self.base_image = torch.zeros(
                (self.config.image_size, self.config.image_size),
                dtype=torch.float32,
                device=self.config.device
            )
        if steps_left is not None:
            self.steps_left = torch.tensor(steps_left, dtype=int, device=self.config.device)
        else:
            self.steps_left = torch.tensor(0, dtype=int, device=self.config.device)

        self.curve_graphic = CurveGraphic2d(
            (self.config.image_size, self.config.image_size),
            self.config.num_bezier_samples,
            self.config.device
        )


    def step(self, action: torch.tensor) -> None:
        bezier_points = action[:-1].reshape((self.config.num_bezier_key_points, 2))
        pass_prob = action[-1]
        
        # TODO: pass_prob

        with torch.no_grad():
            curve_image = self.curve_graphic(
                bezier_points,
                [self.config.bezier_width],
                [self.config.bezier_aa_factor]
            )[0]

        self.base_image += curve_image
        self.base_image = self.base_image.clip(0.0, 1.0)
        self.steps_left -= 1


    def get_observation(self) -> Dict[str, torch.tensor]:
        return {
            "image": self.base_image,
            "steps_left": self.steps_left
        }


    def is_finished(self):
        return self.steps_left >= self.max_steps



if __name__ == "__main__":
    # empty environment
    environment_config = EnvironmentConfig(
        max_steps=10,
        num_bezier_key_points=4
    )
    environment_one = SoloEnvironment(environment_config)
    print(environment_one.get_observation())
    environment_one.step(torch.tensor([
        0.3, 0.3,
        0.5, 0.5,
        0.7, 0.7,
        1.0, 1.0,
        0.0
    ]))

    print(environment_one.get_observation())

    # keep track of images for alternating moves
    alternating_image_stack = []
    alternating_image_stack.append(environment_one.base_image.clone())

    # second environment that builds on the first's image
    environment_two = SoloEnvironment(
        environment_config,
        alternating_image_stack[-1],
        environment_config.max_steps - len(alternating_image_stack)
    )
    print(environment_two.get_observation())
    environment_two.step(torch.tensor([
        0.3, 0.3,
        0.5, 0.5,
        0.7, 0.7,
        1.0, 1.0,
        0.0
    ]))

    print(environment_two.get_observation())
