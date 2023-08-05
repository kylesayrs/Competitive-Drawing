from typing import Optional, Tuple

import os
import torch
from sklearn.metrics import accuracy_score

from competitive_drawing.train.contrastive_learning.config import TrainingConfig
from competitive_drawing.train.contrastive_learning.models import ClassEncoder, ImageEncoder


def load_models(config: TrainingConfig, checkpoint_path: Optional[str]):
    class_encoder = ClassEncoder(config.num_classes, config.latent_size)
    image_encoder = ImageEncoder(config.latent_size, max_temp=config.max_temp)

    if checkpoint_path is not None:
        class_encoder_path = os.path.join(checkpoint_path, "class_encoder.pth")
        image_encoder_path = os.path.join(checkpoint_path, "image_encoder.pth")
        class_encoder.load_state_dict(torch.load(class_encoder_path))
        image_encoder.load_state_dict(torch.load(image_encoder_path))

    return class_encoder, image_encoder


def get_resume_numbers(checkpoint_path: Optional[str]) -> Tuple[str, str]:
    if checkpoint_path is not None:
        basename = os.path.basename(checkpoint_path)
        epoch_num, batch_num = basename.split("_")
        return int(epoch_num), int(batch_num)
    else:
        return 0, 0


def projection_accuracy(
    class_labels: torch.tensor,
    logits: torch.tensor
):
    with torch.no_grad():
        predicted = torch.argmax(logits, dim=1)
        true = torch.argmax(class_labels, dim=1)
        return accuracy_score(true, predicted)
