from typing import Tuple
from pydantic import BaseModel, Field

from competitive_drawing import Settings


class EnvironmentConfig(BaseModel):
    target_image_path: str = Field(default="house_8.png")
    total_num_turns: int = Field(default=1)

    num_bezier_key_points: int = Field(default=3)
    num_bezier_approximations: int = Field(default=10)
    num_bezier_samples: int = Field(default=25)
    bezier_width: float = Field(default=Settings.get("CANVAS_LINE_WIDTH", 1.5))
    bezier_aa_factor: int = Field(default=1.0)
    bezier_length: float = Field(default=Settings.get("IMAGE_SIZE"))

    device: str = Field(default="cpu")


class AgentConfig(BaseModel):
    n_envs: int = Field(default=2)
    total_timesteps: float = Field(default=300_000)

    policy: str = Field(default="MultiInputPolicy")
    policy_kwargs: str = Field(default={})

    learning_rate: float = Field(default=0.0001)
    n_steps: float = Field(default=2, description="The number of steps to run for each environment per update")
    batch_size: int = Field(default=2)
    n_epochs: int = Field(default=128)

    gamma: float = Field(default=0.93)
    gae_lambda: float = Field(default=0.95)
    clip_range: float = Field(default=0.2)

    log_interval: int = Field(default=1)
    n_eval_episodes: int = Field(default=1)
    eval_freq: int = Field(default=128)
    eval_render: bool = Field(default=True)
    progress_bar: bool = Field(default=False)
    verbosity: int = Field(default=2)
    device: str = Field(default="cpu")
    wandb_mode: str = Field(default="disabled")
