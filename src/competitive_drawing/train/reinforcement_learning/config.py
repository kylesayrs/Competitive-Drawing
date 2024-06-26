from typing import Tuple
from pydantic import BaseModel, Field

from competitive_drawing import SETTINGS


class EnvironmentConfig(BaseModel):
    image_shape: Tuple[int, int] = Field(default=(50, 50))
    total_num_turns: int = Field(default=SETTINGS.total_num_turns)

    num_bezier_key_points: int = Field(default=4)
    num_bezier_approximations: int = Field(default=10)
    num_bezier_samples: int = Field(default=15)
    bezier_width: float = Field(default=SETTINGS.canvas_line_width)
    bezier_aa_factor: int = Field(default=1.0)
    bezier_length: float = Field(default=(
        SETTINGS.distance_per_turn /
        SETTINGS.canvas_size *
        SETTINGS.image_size
    ))

    step_reward_factor: float = Field(default=0)

    device: str = Field(default="cpu")


class AgentConfig(BaseModel):
    n_envs: int = Field(default=2)
    total_timesteps: float = Field(default=300_000)

    policy: str = Field(default="MultiInputPolicy")
    policy_kwargs: str = Field(default={})

    learning_rate: float = Field(default=0.0005)
    n_steps: float = Field(default=1024, description="The number of steps to run for each environment per update")
    batch_size: int = Field(default=64)
    n_epochs: int = Field(default=15)

    gamma: float = Field(default=0.93)
    gae_lambda: float = Field(default=0.95)
    clip_range: float = Field(default=0.2)

    log_interval: int = Field(default=1)
    progress_bar: bool = Field(default=False)
    verbosity: int = Field(default=2)
    device: str = Field(default="cpu")
    wandb_mode: str = Field(default="disabled")
