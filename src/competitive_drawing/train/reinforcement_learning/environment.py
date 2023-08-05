from typing import Dict

import torch

from competitive_drawing.train.contrastive_learning import ClassEncoder, ImageEncoder
from competitive_drawing.train.contrastive_learning import TrainingConfig as ClConfig
from competitive_drawing.train.reinforcement_learning import EnvironmentConfig
from competitive_drawing.diff_graphics import CurveGraphic2d


class SoloEnvironment():
    def __init__(
        self,
        environment_config: EnvironmentConfig,
    ):
        super().__init__()
        self.config = environment_config

        self.curve_graphic = CurveGraphic2d(
            self.config.image_shape,
            self.config.num_bezier_samples,
            self.config.bezier_length,
            self.config.device
        )

        self.class_encoder = ClassEncoder(ClConfig.latent_size, ClConfig.num_classes)
        self.class_encoder.eval()
        self.image_encoder = ImageEncoder(ClConfig.latent_size, max_temp=0.0)
        self.image_encoder.eval()

        self.class_embeddings = self._get_class_embeddings()

        self.image = None
        self.steps_left = None
        self.target_embedding = None
        self.reset()


    def _get_class_embeddings(self):
        return self.class_encoder(torch.eye(ClConfig.num_classes))
    

    def reset(self):
        self.image = torch.zeros(
            self.config.image_shape,
            dtype=torch.float32,
            device=self.config.device
        )

        self.steps_left = torch.tensor(0, dtype=int, device=self.config.device)

        # TODO: some random way of selecting
        self.target_embedding = self.class_embeddings[0]


    def step(self, action: torch.tensor) -> None:
        bezier_points = action.reshape((self.config.num_bezier_key_points, 2))
        
        with torch.no_grad():
            curve_image = self.curve_graphic(
                bezier_points,
                [self.config.bezier_width],
                [self.config.bezier_aa_factor]
            )

        self.image += curve_image
        self.steps_left -= 1


    def get_observation(self) -> Dict[str, torch.tensor]:
        return {
            "image": self.image,
            "steps_left": self.steps_left,
            "target_embedding": self.target_embedding,
        }
    

    def get_reward(self) -> float:
        embedding = self.image_encoder(self.image)
        return embedding @ self.target_embedding.T


    def is_finished(self):
        return self.steps_left >= self.max_steps
