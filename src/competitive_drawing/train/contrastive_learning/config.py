from typing import Tuple, List
from pydantic import BaseModel, Field
from .classes import class_names as train_class_names


class ModelsConfig(BaseModel):
    class_names: List[str] = Field(default=train_class_names)
    num_classes: int = Field(default=len(train_class_names))  #Field(default=345)
    latent_size: int = Field(default=32)  #Field(default=128)
    max_temp: float = Field(default=100.0)

    images_dir: str = Field(default="images")
    image_shape: Tuple[int, int] = Field(default=(50, 50))

    num_epochs: int = Field(default=3)
    class_lr: float = Field(default=1e-2)
    image_lr: float = Field(default=1e-4)
    batch_size: int = Field(default=256)
    test_batch_size: int = Field(default=256)
    test_size: float = Field(default=0.15)

    log_freq: int = Field(default=10, description="Training batches per log")
    save_freq: int = Field(default=1000, description="Training batches per save")

    device: str = Field(default="cuda")
    wandb_mode: str = Field(default="online")
