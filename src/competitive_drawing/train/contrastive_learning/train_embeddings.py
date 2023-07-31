import os
import wandb
import torch
from sklearn.model_selection import train_test_split

from competitive_drawing.train.contrastive_learning.config import TrainingConfig
from competitive_drawing.train.utils import (
    get_all_local_labels, load_data, QuickDrawDataset, RandomResizePad
)
from competitive_drawing.train.contrastive_learning.models import ClassEncoder, ImageEncoder


def train_models(config: TrainingConfig):
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

    train_dataset = QuickDrawDataset(x_train, y_train, is_test=False)
    test_dataset = QuickDrawDataset(x_test, y_test, is_test=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,
                                               shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size,
                                              shuffle=True, num_workers=0, drop_last=True)

    # create models
    class_encoder = ClassEncoder(config.num_classes, config.latent_size)
    image_encoder = ImageEncoder(config.latent_size)

    class_optimizer = torch.optim.Adam(class_encoder.parameters(), lr=config.class_lr)
    image_optimizer = torch.optim.Adam(image_encoder.parameters(), lr=config.image_lr)

    criterion = torch.nn.CosineEmbeddingLoss(reduction="none")
    train_target = torch.ones((config.batch_size, ), device=config.device)
    test_target = torch.ones((config.test_batch_size, ), device=config.device)

     # train
    print("Begin training")
    metrics = {}
    for epoch_index in range(config.num_epochs):

        running_loss = 0.0
        for batch_index, (images, labels) in enumerate(train_loader):
            # augmentations
            images = random_resize_pad(images)

            # to device
            labels = labels.to(config.device)
            images = images.to(config.device)

            # zero the parameter gradients
            class_optimizer.zero_grad()
            image_optimizer.zero_grad()

            # forward + backward + optimize
            class_embedding = class_encoder(labels)
            image_embedding = image_encoder(images)
            loss = criterion(class_embedding, image_embedding, train_target).sum()
            loss.backward()
            class_optimizer.step()
            image_optimizer.step()

            # logging
            running_loss += loss.item()
            if batch_index % config.log_freq == 0:
                with torch.no_grad():
                    test_images, test_labels = next(iter(test_loader))
                    test_labels = test_labels.to(config.device)
                    test_images = test_images.to(config.device)

                    test_class_embedding = class_encoder(test_labels)
                    test_image_embedding = image_encoder(test_images)

                    test_loss = criterion(test_class_embedding, test_image_embedding, test_target).sum()

                running_train_loss = running_loss / config.log_freq
                running_loss = 0.0

                print(
                    f"[{epoch_index + 1}, {batch_index + 1:5d}] "
                    f"train_loss: {running_train_loss} "
                    f"test_loss: {test_loss.item()}"
                )
                metrics = {
                    "train_loss": running_train_loss,
                    "test_loss": test_loss.item()
                }
                wandb.log(metrics)

            # save model checkpoint
            if batch_index % config.save_freq == 0:
                save_dir = os.path.join("checkpoints", run.id, "{epoch_index}_{batch_index}")
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
    training_config = TrainingConfig()
    train_models(training_config)
