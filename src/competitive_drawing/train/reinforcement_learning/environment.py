from typing import Dict

import cv2
import torch
import numpy
from gym import Env, spaces

from competitive_drawing.train.contrastive_learning import (
    ClassEncoder, ImageEncoder, ModelsConfig
)
from competitive_drawing.train.reinforcement_learning import EnvironmentConfig
from competitive_drawing.diff_graphics import CurveGraphic2d


class StrokeEnvironment(Env):
    def __init__(
        self,
        environment_config: EnvironmentConfig,
        models_config: ModelsConfig,
    ):
        super().__init__()
        self.config = environment_config
        self.models_config = models_config

        self.class_encoder = ClassEncoder(models_config.num_classes, models_config.latent_size)
        self.image_encoder = ImageEncoder(models_config.latent_size, max_temp=0.0)
        self._freeze_and_eval(self.class_encoder)
        self._freeze_and_eval(self.image_encoder)

        self.class_embeddings = self._get_class_embeddings()

        self.curve_graphic = CurveGraphic2d(
            self.config.image_shape,
            self.config.num_bezier_samples,
            self.config.bezier_length,
            self.config.device
        )

        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()

        self.image = None
        self.steps_left = None
        self.target_embedding = None
        self.reset()


    def _make_observation_space(self):
        return spaces.Dict({
            "image": spaces.Box(0.0, 1.0, self.models_config.image_shape),
            "steps_left": spaces.Box(0.0, self.config.total_num_turns, (1, )),
            "target": spaces.Box(0.0, 1.0, (self.models_config.latent_size, )),
        })


    def _make_action_space(self):
        return spaces.Box(0.0, 1.0, (self.config.num_bezier_key_points, 2))


    def _get_class_embeddings(self):
        return self.class_encoder(torch.eye(self.models_config.num_classes))
    

    def _freeze_and_eval(self, model: torch.nn.Module):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
    

    def reset(self):
        self.image = torch.zeros(
            self.config.image_shape,
            dtype=torch.float32,
            device=self.config.device
        )

        self.steps_left = torch.tensor(self.config.total_num_turns, dtype=int, device=self.config.device)

        class_one, class_two = numpy.random.choice(self.models_config.num_classes, 2)
        t = torch.rand(1)[0]  # bias towards edges maybe

        theta_one = torch.arccos(self.class_embeddings[class_one])
        theta_two = torch.arccos(self.class_embeddings[class_two])
        theta_target = torch.lerp(theta_one, theta_two, t)
        self.target = torch.cos(theta_target)

        return self.get_observation()


    def step(self, action: torch.tensor) -> None:       
        cv2.polylines(self.image, action, False, 1.0)
        print(self.image)
        self.steps_left -= 1

        observation = self.get_observation()
        reward = self.get_reward()
        is_finished = self.is_finished()
        info = {}

        return observation, reward, is_finished, info


    def get_observation(self) -> Dict[str, torch.tensor]:
        return {
            "image": self.image,
            "steps_left": self.steps_left,
            "target": self.target,
        }
    

    def get_reward(self) -> float:
        if self.config.step_reward_factor > 0.0:
            embedding = self.image_encoder(self.image)
            return embedding @ self.target.T
        else:
            return 0.0


    def is_finished(self):
        return self.steps_left >= self.config.total_num_turns
