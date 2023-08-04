from typing import Dict, Any

import os
import torch
import numpy
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from competitive_drawing.train.contrastive_learning.config import TrainingConfig
from competitive_drawing.train.utils import load_data, to_one_hot
from competitive_drawing.train.contrastive_learning.models import (
    ClassEncoder, ImageEncoder
)


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path")
parser.add_argument("--perplexity", type=int, default=2)
parser.add_argument("--images_per_class", type=int, default=2)


def validate_models(
    config: TrainingConfig,
    args: Dict[str, Any]
):
    class_encoder_path = os.path.join(args.checkpoint_path, "class_encoder.pth")
    class_encoder = ClassEncoder(config.num_classes, config.latent_size)
    class_encoder.load_state_dict(torch.load(class_encoder_path))
    class_encoder.eval()

    image_encoder_path = os.path.join(args.checkpoint_path, "image_encoder.pth")
    image_encoder = ImageEncoder(config.latent_size)
    image_encoder.load_state_dict(torch.load(image_encoder_path))
    image_encoder.eval()

    all_images, all_labels, label_names = load_data(
        config.images_dir, config.image_shape, one_hot=False
    )
    num_classes = len(label_names)

    class_encodings = []
    image_encodings = []

    with torch.no_grad():
        class_encodings = class_encoder(torch.eye(num_classes)).tolist()

    image_labels = []
    for label in range(num_classes):
        image_indexes = [index for index, _label in enumerate(all_labels) if _label == label]
        for image_index in image_indexes[:args.images_per_class]:
            image = numpy.array(all_images[image_index]).reshape((1, 1, 50, 50))
            with torch.no_grad():
                image_encoding = image_encoder(torch.tensor(image, dtype=torch.float32))[0].tolist()
            image_encodings.append(image_encoding)
            image_labels.append(label_names[all_labels[image_index]])

    embeddings = numpy.array(class_encodings + image_encodings)
    if config.latent_size > 2:
        tsne_model = TSNE(
            n_components=2,
            learning_rate="auto",
            init="random",
            perplexity=args.perplexity
        )
        embeddings = tsne_model.fit_transform(embeddings)
    class_embeddings = embeddings[:len(class_encodings)]
    image_embeddings = embeddings[len(class_encodings):]

    plt.scatter(*class_embeddings.T, color="red")
    for point, label in zip(class_embeddings, label_names):
        plt.annotate(label, point)

    plt.scatter(*image_embeddings.T, color="blue")
    for point, label in zip(image_embeddings, image_labels):
        plt.annotate(label, point)

    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    training_config = TrainingConfig()

    validate_models(training_config, args)
