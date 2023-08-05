from typing import Optional, Dict

import torch

from competitive_drawing.train.reinforcement_learning import EnvironmentConfig
from competitive_drawing.diff_graphics import CurveGraphic2d


class SoloEnvironment():
    def __init__(
        self,
        environment_config: EnvironmentConfig,
    ):
        super().__init__()
        self.config = environment_config

        self.base_image = torch.zeros(
            self.config.image_shape,
            dtype=torch.float32,
            device=self.config.device
        )
        self.steps_left = torch.tensor(0, dtype=int, device=self.config.device)

        self.curve_graphic = CurveGraphic2d(
            self.config.image_shape,
            self.config.num_bezier_samples,
            self.config.device
        )


    def step(self, action: torch.tensor) -> None:
        bezier_points = action[:-2].reshape((self.config.num_bezier_key_points, 2))
        pass_prob = action[-1]
        
        # TODO: pass_prob

        with torch.no_grad():
            curve_image = self.curve_graphic(
                bezier_points,
                [self.config.bezier_width],
                [self.config.bezier_aa_factor]
            )

        self.base_image += curve_image
        self.steps_left -= 1


    def get_observation(self) -> Dict[str, torch.tensor]:
        return {
            "image": self.base_image,
            "steps_left": self.steps_left
        }


    def is_finished(self):
        return self.steps_left >= self.max_steps
