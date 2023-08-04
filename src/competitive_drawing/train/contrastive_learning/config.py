from typing import Tuple
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    num_classes: int = Field(default=2)  #Field(default=345)
    latent_size: int = Field(default=2)  #Field(default=128)

    images_dir: str = Field(default="images")
    image_shape: Tuple[int, int] = Field(default=(50, 50))
    resize_scale: Tuple[float, float] = Field(default=(0.2, 1.0))

    temperature: float = Field(default=1.0)

    num_epochs: int = Field(default=10)
    class_lr: float = Field(default=1e-1)
    image_lr: float = Field(default=1e-3)
    batch_size: int = Field(default=256)
    test_batch_size: int = Field(default=256)
    test_size: float = Field(default=0.15)

    log_freq: int = Field(default=2, description="Training batches per log")
    save_freq: int = Field(default=200, description="Training batches per save")

    device: str = Field(default="cpu")
    wandb_mode: str = Field(default="disabled")
