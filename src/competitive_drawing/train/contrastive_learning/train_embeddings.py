from typing import Dict, Any

import os
import wandb
import torch
import argparse
from sklearn.model_selection import train_test_split
from sparseml.pytorch.optim import ScheduledModifierManager

from competitive_drawing.train.contrastive_learning.config import TrainingConfig
from competitive_drawing.train.utils import load_data, QuickDrawDataset, RandomResizePad
from competitive_drawing.train.contrastive_learning.utils import (
    load_models, get_resume_numbers, projection_accuracy
)


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--class_recipe_path", type=str, default=None)
parser.add_argument("--image_recipe_path", type=str, default=None)

def train_models(config: TrainingConfig, args: Dict[str, Any]):
    # wandb config
    run = wandb.init(
        project="competitive_drawing_contrastive",
        entity="kylesayrs",
        mode=config.wandb_mode,
        config=config.dict()
    )
    print(wandb.config)

    # load data
    all_images, all_labels, label_names = load_data(
        config.images_dir, config.image_shape, one_hot=True
    )

    num_classes = len(label_names)
    if (num_classes != config.num_classes):
        raise ValueError(
            f"Warning: config specified {config.num_classes} classes, but only "
            f"found {num_classes}"
        )

    # image augmentation
    random_resize_pad = RandomResizePad(
        config.image_shape, scale=config.resize_scale, value=0,
    )

    # create datasets
    x_train, x_test, y_train, y_test = train_test_split(
        all_images,
        all_labels,
        test_size=config.test_size,
        shuffle=True,
    )
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    classes = torch.eye(len(label_names))
    train_dataset = QuickDrawDataset(x_train, y_train, is_test=False)
    test_dataset = QuickDrawDataset(x_test, y_test, is_test=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,
                                               shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size,
                                              shuffle=True, num_workers=0, drop_last=True)

    # create models and optimizers
    class_encoder, image_encoder = load_models(config, args.checkpoint_path)

    class_optimizer = torch.optim.Adam(class_encoder.parameters(), lr=config.class_lr)
    image_optimizer = torch.optim.Adam(image_encoder.parameters(), lr=config.image_lr)

    if args.class_recipe_path is not None:
        manager = ScheduledModifierManager.from_yaml(args.class_recipe_path)
        class_optimizer = manager.modify(class_encoder, class_optimizer, len(train_loader))
    if args.image_recipe_path is not None:
        manager = ScheduledModifierManager.from_yaml(args.image_recipe_path)
        image_optimizer = manager.modify(image_encoder, image_optimizer, len(train_loader))

    criterion = torch.nn.CrossEntropyLoss()

    # train
    print("Begin training")

    resume_epoch, resume_batch = get_resume_numbers(args.checkpoint_path)
    metrics = {}
    for epoch_index in range(config.num_epochs):
        epoch_index += resume_epoch
        for batch_index, (images, labels) in enumerate(train_loader):
            batch_index += resume_batch
            class_encoder.train()
            image_encoder.train()

            # augmentations
            images = random_resize_pad(images)

            # to device
            labels = labels.to(config.device)
            images = images.to(config.device)

            # zero the parameter gradients
            class_optimizer.zero_grad()
            image_optimizer.zero_grad()

            # forward
            class_embedding = class_encoder(classes)
            image_embedding = image_encoder(images)

            # calculate loss
            logits  = image_embedding @ class_embedding.T
            loss = criterion(logits, labels)
            accuracy = projection_accuracy(labels, logits)
            
            # optimize
            loss.backward()
            class_optimizer.step()
            image_optimizer.step()

            # logging
            if batch_index % config.log_freq == 0:
                class_encoder.eval()
                image_encoder.eval()
                with torch.no_grad():
                    test_images, test_labels = next(iter(test_loader))
                    test_labels = test_labels.to(config.device)
                    test_images = test_images.to(config.device)

                    test_class_embedding = class_encoder(classes)
                    test_image_embedding = image_encoder(test_images)

                    test_logits = test_image_embedding @ test_class_embedding.T
                    test_loss = criterion(test_logits, test_labels)

                    test_accuracy = projection_accuracy(test_labels, test_logits)

                metrics = {
                    "train_loss": loss.item(),
                    "train_accuracy": accuracy,
                    "test_loss": test_loss.item(),
                    "test_accuracy": test_accuracy,
                    "temp": image_encoder.temperature.item()
                }

                print(f"[{epoch_index}, {batch_index + 1:5d}]", end=" ")
                for metric, value in metrics.items():
                    print(f"{metric}: {value:0.3f}", end=" ")
                print()

                wandb.log(metrics)

            # save model checkpoint
            if batch_index % config.save_freq == 0:
                save_dir = os.path.join(
                    "checkpoints", run.id, f"{epoch_index}_{batch_index}"
                )
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    class_encoder.state_dict(),
                    os.path.join(save_dir, "class_encoder.pth")
                )
                torch.save(
                    image_encoder.state_dict(),
                    os.path.join(save_dir, "image_encoder.pth")
                )
                print(f"Saved models to {save_dir}")

    print("Finished Training")


if __name__ == "__main__":
    args = parser.parse_args()

    training_config = TrainingConfig()
    train_models(training_config, args)
