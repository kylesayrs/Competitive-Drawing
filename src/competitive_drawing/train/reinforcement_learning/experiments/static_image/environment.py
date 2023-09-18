from typing import Dict

import cv2
import torch
import numpy
from gym import Env, spaces

from competitive_drawing.train.reinforcement_learning import EnvironmentConfig
from competitive_drawing.diff_graphics import CurveGraphic2d


class StrokeEnvironment(Env):
    def __init__(self, environment_config: EnvironmentConfig):
        super().__init__()
        self.config = environment_config

        self.target_image = cv2.imread(self.config.target_image_path, cv2.IMREAD_GRAYSCALE)

        self.curve_graphic = CurveGraphic2d(
            self.target_image.shape,
            self.config.num_bezier_samples,
            self.config.bezier_length,
            self.config.device
        )

        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()

        self.image = None
        self.steps_left = None
        self.reset()


    def _make_observation_space(self):
        return spaces.Dict({
            "image": spaces.Box(0.0, 1.0, self.target_image.shape),
            "steps_left": spaces.Box(0.0, self.config.total_num_turns, (1, )),
        })


    def _make_action_space(self):
        return spaces.Box(0.0, 1.0, (self.config.num_bezier_key_points, 2))
    

    def reset(self):
        self.image = torch.zeros(
            self.target_image.shape,
            dtype=torch.float32,
        )

        self.steps_left = torch.tensor(self.config.total_num_turns, dtype=int, device=self.config.device)

        return self.get_observation()


    def step(self, action: numpy.ndarray) -> None:       
        action = torch.tensor(action)
        self.image += self.curve_graphic(
            action, [self.config.bezier_width], [self.config.bezier_aa_factor]
        )[0]
        self.steps_left -= 1

        observation = self.get_observation()
        reward = self.get_reward()
        is_finished = self.is_finished()
        info = {}

        return observation, reward, is_finished, info


    def get_observation(self) -> Dict[str, torch.tensor]:
        return {
            "image": self.image,
            "steps_left": self.steps_left
        }
    

    def get_reward(self) -> float:
        print(self.target_image)
        print(self.image)
        with torch.no_grad():
            return torch.mse(self.target_image, self.image)


    def is_finished(self):
        return self.steps_left >= self.config.total_num_turns
