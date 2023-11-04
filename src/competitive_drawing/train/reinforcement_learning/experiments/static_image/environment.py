from typing import Dict, Optional

import os
import cv2
import torch
import numpy
from gymnasium import Env, spaces

from competitive_drawing.train.reinforcement_learning import EnvironmentConfig
from competitive_drawing.diff_graphics import CurveGraphic2d
from competitive_drawing.diff_graphics.utils import draw_output_and_target


class StrokeEnvironment(Env):
    def __init__(self, environment_config: EnvironmentConfig):
        super().__init__()
        self.config = environment_config
        self.render_mode = "cv2"

        self.target_images = [
            torch.tensor(
                cv2.imread(os.path.join(self.config.target_images_dir, file_name), cv2.IMREAD_GRAYSCALE) / 255
            )
            for file_name in os.listdir(self.config.target_images_dir)
        ]

        self.curve_graphic = CurveGraphic2d(
            self.target_images[0].shape,
            self.config.num_bezier_samples,
            self.config.bezier_length,
            self.config.device,
        )

        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()

        self.target_image = None
        self.image = None
        self.steps_left = None
        self.reset()


    def _get_target_image(self):
        #return self.target_images[numpy.random.randint(len(self.target_images))]
        return self.target_images[0]


    def _make_observation_space(self):
        return spaces.Dict({
            "image": spaces.Box(0.0, 1.0, (1, )),
            #"image": spaces.Box(0.0, 1.0, self.target_images[0].shape),
            #"steps_left": spaces.Box(0.0, self.config.total_num_turns, (1, )),
        })


    def _make_action_space(self):
        return spaces.Box(0.0, 1.0, (self.config.num_bezier_key_points, 2))
    

    def reset(self, seed: Optional[int] = 0):
        self.target_image = self._get_target_image()

        self.image = torch.zeros(
            self.target_image.shape,
            dtype=torch.float32,
        )

        self.steps_left = torch.tensor(self.config.total_num_turns, dtype=int, device=self.config.device)

        return self.get_observation(), {}


    def step(self, action: numpy.ndarray) -> None:
        action = torch.tensor(action)
        self.image += self.curve_graphic(
            action, [self.config.bezier_width], [self.config.bezier_aa_factor]
        )[0]
        self.image = torch.clamp(self.image, 0.0, 1.0)
        self.steps_left -= 1

        observation = self.get_observation()
        reward = self.get_reward()
        print(reward)
        is_finished = self.is_finished()
        truncated = False
        info = {}

        return observation, reward, is_finished, truncated, info


    def get_observation(self) -> Dict[str, torch.tensor]:
        return {
            "image": numpy.array([0]),
            #"image": self.image,
            #"steps_left": self.steps_left
        }
    

    def get_reward(self) -> float:
        with torch.no_grad():
            loss = torch.nn.functional.mse_loss(self.image, self.target_image)
            return 1.0 - loss.item()


    def is_finished(self):
        return self.steps_left <= 0


    def render(self):
        if self.render_mode == "cv2":
            render_image = draw_output_and_target(self.image, self.target_image)
            cv2.imshow("render", render_image * 255)
            cv2.waitKey(0)
        else:
            raise ValueError(f"Unknown render mode {self.render_mode}")


    def close(self):
        cv2.destroyAllWindows()
