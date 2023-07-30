from typing import Dict

import torch

from competitive_drawing.train.rl_gan import EnvironmentConfig, Critic
from competitive_drawing.diff_graphics import CurveGraphic2d


class AlternatingSoloEnvironment():
    def __init__(
        self,
        environment_config: EnvironmentConfig,
        agent_number: int,
        critic: Critic
    ):
        super().__init__()
        self.config = environment_config
        self.agent_number = agent_number
        self.critic = critic

        self.curve_graphic = CurveGraphic2d(
            (self.config.image_size, self.config.image_size),
            self.config.num_bezier_samples,
            self.config.device
        )

        self.image = None
        self.steps_left = None
        self.reset()


    def reset(self) -> None:
        self.image = torch.zeros(
            (self.config.image_size, self.config.image_size),
            dtype=torch.float32,
            device=self.config.device
        )

        self.steps_left = torch.tensor(0, dtype=int, device=self.config.device)

        return self.get_observation()


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

        self.image += curve_image
        self.image = self.image.clip(0.0, 1.0)
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
    

    def get_image(self) -> torch.tensor:
        return self.image
    

    def get_reward(self) -> torch.tensor:
        if environment_config.shaped_reward:
            with torch.no_grad():
                critic_scores = self.critic(self.image)
                real_score = critic_scores[2 * (self.agent_number - 1)]

        return real_score


    def is_finished(self) -> bool:
        return self.steps_left >= self.config.max_steps



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
