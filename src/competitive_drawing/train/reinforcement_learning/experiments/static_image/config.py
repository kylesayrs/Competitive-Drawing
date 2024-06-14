from typing import Tuple, Optional
from pydantic import BaseModel, Field

import torch

from competitive_drawing import SETTINGS


class EnvironmentConfig(BaseModel):
    target_images_dir: str = Field(default="houses")
    total_num_turns: int = Field(default=3)

    num_bezier_key_points: int = Field(default=3)
    num_bezier_approximations: int = Field(default=10)
    num_bezier_samples: int = Field(default=25)
    bezier_width: float = Field(default=SETTINGS.canvas_line_width)
    bezier_aa_factor: int = Field(default=5.0)
    bezier_length: float = Field(default=SETTINGS.image_size)

    device: str = Field(default="cpu")

class ModelConfig(BaseModel):
    n_envs: int = Field(default=2)

    policy: str = Field(default="MultiInputPolicy")
    policy_kwargs: str = Field(default={
        "activation_fn": torch.nn.ReLU
    })

    log_interval: int = Field(default=20, description="episodes per log")
    n_eval_episodes: int = Field(default=1)
    eval_freq: int = Field(default=1000, description="steps per evaluation")
    eval_render: bool = Field(default=True)

    progress_bar: bool = Field(default=True)
    verbose: int = Field(default=2)
    tensorboard_log: str = Field(default="./tensorboard")
    wandb_mode: str = Field(default="disabled")
    device: str = Field(default="cpu")

    class Config:
        arbitrary_types_allowed = True


class DDPGConfig(ModelConfig):
    total_timesteps: float = Field(default=100_000)

    learning_starts: int = Field(default=128)
    learning_rate: float = Field(default=3e-5)
    train_freq: Tuple[int, str] = Field(default=(10, "step"))
    batch_size: int = Field(default=64)
    gamma: float = Field(default=0.9)

    action_noise: Optional[str] = Field(default="normal")
    action_noise_mu: float = Field(default=0.0)
    action_noise_sigma: float = Field(default=0.1)

    buffer_size: int = Field(default=100_000)
    optimize_memory_usage: bool = Field(default=False)
