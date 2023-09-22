from typing import Tuple, Optional
from pydantic import BaseModel, Field

import torch

from competitive_drawing import Settings


class EnvironmentConfig(BaseModel):
    target_image_path: str = Field(default="house_8.png")
    total_num_turns: int = Field(default=5)

    num_bezier_key_points: int = Field(default=3)
    num_bezier_approximations: int = Field(default=10)
    num_bezier_samples: int = Field(default=25)
    bezier_width: float = Field(default=Settings.get("CANVAS_LINE_WIDTH", 1.5))
    bezier_aa_factor: int = Field(default=1.0)
    bezier_length: float = Field(default=Settings.get("IMAGE_SIZE"))

    device: str = Field(default="cpu")

class ModelConfig(BaseModel):
    n_envs: int = Field(default=2)

    policy: str = Field(default="MultiInputPolicy")
    policy_kwargs: str = Field(default={
        "activation_fn": torch.nn.ReLU
    })

    log_interval: int = Field(default=20, description="episodes per log")
    n_eval_episodes: int = Field(default=1)
    eval_freq: int = Field(default=1_000, description="steps per evaluation")
    eval_render: bool = Field(default=True)

    progress_bar: bool = Field(default=True)
    verbose: int = Field(default=2)
    tensorboard_log: str = Field(default="./tensorboard")
    wandb_mode: str = Field(default="disabled")
    device: str = Field(default="cpu")

    class Config:
        arbitrary_types_allowed = True


class PPOConfig(ModelConfig):
    total_timesteps: float = Field(default=300_000)

    learning_rate: float = Field(default=0.5e-6)
    n_steps: float = Field(default=32, description="The number of steps to run for each environment per update")
    batch_size: int = Field(default=2)
    n_epochs: int = Field(default=128)

    gamma: float = Field(default=0.93)
    gae_lambda: float = Field(default=0.95)
    clip_range: float = Field(default=0.2)


class DDPGConfig(ModelConfig):
    total_timesteps: float = Field(default=10_000)

    learning_starts: int = Field(default=128)
    learning_rate: float = Field(default=3e-2)
    train_freq: Tuple[int, str] = Field(default=(10, "step"))
    batch_size: int = Field(default=64)
    gamma: float = Field(default=0.9)

    action_noise: Optional[str] = Field(default="normal")
    action_noise_mu: float = Field(default=0.0)
    action_noise_sigma: float = Field(default=0.1)

    buffer_size: int = Field(default=100_000)
    optimize_memory_usage: bool = Field(default=False)
