from typing import Tuple
from pydantic import BaseModel, Field

from competitive_drawing import Settings


class EnvironmentConfig(BaseModel):
    image_size: Tuple[int, int] = Field(default=Settings.get("IMAGE_SIZE", 50))
    max_num_turns: int = Field(default=Settings.get("TOTAL_NUM_TURNS", 10))

    num_bezier_key_points: int = Field(default=4)
    num_bezier_approximations: int = Field(default=10)
    num_bezier_samples: int = Field(default=15)
    bezier_width: float = Field(default=Settings.get("CANVAS_LINE_WIDTH", 1.5))
    bezier_aa_factor: int = Field(default=1.0)

    shaped_reward: bool = Field(default=False)

    device: str = Field(default="cpu")


class TrainingConfig(BaseModel):
    num_episodes: int = Field(default=10)

    log_interval: int = Field(default=128, description="Episodes per log")


class AgentConfig(BaseModel):
    policy: str = Field(default="")
    gradient_steps: int = Field(default=128)
    batch_size: int = Field(default=128)
    pass


class CriticConfig(BaseModel):
    logit_norm: bool = Field(default=False)
