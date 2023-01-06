from typing import List, Optional, Tuple, Dict

import os
import numpy
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from timm.data import Mixup

from utils import load_data, QuickDrawDataset, Classifier, RandomResizePad, upload_model

DEVICE = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


def train_model(
    class_names: List[str],
    data_dir: str,
    num_epochs: int = 6,
    batch_size: int = 128,
    test_batch_size: int = 128,
    test_size: float = 0.2,
    lr: float = 0.005,
    optimizer: str = "Adam",
    momentum: float = 0.9,
    image_shape: Tuple[int, int] = (50, 50),
    mixup_alpha: float = 0.3,
    cutmix_alpha: float = 1.0,
    cutmix_prob: float = 0.8,
    label_smoothing: float = 0.1,
    resize_scale: Tuple[float, float] = (0.2, 1.0),
    patience_length: Optional[int] = 3,
    patience_threshold: Optional[float] = 0.95,
    logging_rate: int = 1000,
    save_checkpoints: bool = False,
    model_name: Optional[str] = None,
    wandb_mode: str = "online",
):
    assert class_names[0] < class_names[1], "class names are out of order!"

    # wandb
    model_name = model_name or "-".join(class_names)
    run = wandb.init(
        project="competitive_drawing",
        entity="kylesayrs",
        name=model_name,
        reinit=True,
        mode=wandb_mode,
        config={
            "model_name": model_name,
            "image_shape": image_shape,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "num_classes": len(class_names),
            "lr": lr,
            "optimizer": optimizer,
            "momentum": momentum,
            "patience_length": patience_length,
            "patience_threshold": patience_threshold,
            "logging_rate": logging_rate,
            "test_batch_size": test_batch_size,
            "test_size": test_size,
            "mixup_alpha": mixup_alpha,
            "cutmix_alpha": cutmix_alpha,
            "cutmix_prob": cutmix_prob,
            "label_smoothing": label_smoothing,
            "resize_scale": resize_scale,
            "save_checkpoints": save_checkpoints,
            "data_dir": data_dir,
        }
    )
    print(wandb.config)

    # transforms
    train_mixup = Mixup(
        mixup_alpha=wandb.config["mixup_alpha"],
        cutmix_alpha=wandb.config["cutmix_alpha"],
        prob=wandb.config["cutmix_prob"],
        label_smoothing=wandb.config["label_smoothing"],
        num_classes=wandb.config["num_classes"],
    )
    random_resize_pad = RandomResizePad(
        wandb.config["image_shape"],
        scale=wandb.config["resize_scale"],
        value=0,
    )

    # load data
    all_images, all_labels, label_names = load_data(
        wandb.config["data_dir"],
        wandb.config["image_shape"],
        class_names=class_names
    )

    # create datasets
    x_train, x_test, y_train, y_test = train_test_split(
        all_images,
        all_labels,
        test_size=wandb.config["test_size"],
        shuffle=True,
    )
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    train_dataset = QuickDrawDataset(x_train, y_train, is_test=False)
    test_dataset = QuickDrawDataset(x_test, y_test, is_test=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config["batch_size"],
                                              shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=wandb.config["test_batch_size"],
                                              shuffle=True, num_workers=0, drop_last=True)

    model = Classifier(wandb.config["num_classes"]).to(DEVICE)
    if wandb.config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config["lr"], momentum=wandb.config["momentum"])
    elif wandb.config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["lr"])
    else:
        raise ValueError(f"Unknown optimizer {wandb.config['optimizer']}")
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    #print(model)

    # train
    print("begin training")
    epoch_test_accs = []
    test_accuracy = 0.0
    metrics = {}
    for epoch in range(wandb.config["num_epochs"]):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (images, raw_labels) in enumerate(train_loader):
            # mixup/ cutmix
            images, cutmix_labels = train_mixup(images, raw_labels)
            images = random_resize_pad(images)

            # to device
            images = images.to(DEVICE)
            cutmix_labels = cutmix_labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            _, outputs = model(images)
            loss = criterion(outputs, cutmix_labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % wandb.config["logging_rate"] == wandb.config["logging_rate"] - 1:
                train_accuracy = accuracy_score(raw_labels.cpu(), numpy.argmax(outputs.cpu().detach().numpy(), axis=1))

                with torch.no_grad():
                    test_images, test_labels = next(iter(test_loader))
                    test_images = random_resize_pad(test_images)
                    test_images = test_images.to(DEVICE)
                    test_labels = test_labels.to(DEVICE)
                    _, test_outputs = model(test_images)
                    test_accuracy = accuracy_score(test_labels.cpu(), numpy.argmax(test_outputs.cpu(), axis=1))
                    test_loss = criterion(test_outputs, test_labels)

                print(
                    f"[{epoch + 1}, {i + 1:5d}] "
                    f"loss: {running_loss / wandb.config['logging_rate']} "
                    f"test_acc: {test_accuracy}"
                )
                metrics = {
                    "loss": running_loss / wandb.config["logging_rate"],
                    "test_loss": test_loss.item(),
                    "train_acc": train_accuracy,
                    "test_acc": test_accuracy,
                }
                wandb.log(metrics)

                running_loss = 0.0

        # check patience each epoch
        epoch_test_accs.append(test_accuracy)
        if (
            wandb.config["patience_length"] and wandb.config["patience_threshold"] and
            len(epoch_test_accs) >= wandb.config["patience_length"] and
            all([
                acc >= wandb.config["patience_threshold"]
                for acc in epoch_test_accs[-1 * wandb.config["patience_length"]:]
            ])
        ):
            break


        # save each epoch
        if wandb.config["save_checkpoints"]:
            os.makedirs(f"./checkpoints/{run.id}", exist_ok=True)
            print(f"saving model to ./checkpoints/{run.id}/epoch{epoch}.pth")
            torch.save(model.state_dict(), f"./checkpoints/{run.id}/epoch{epoch}.pth")

    print("Finished Training")
    upload_model(model, wandb.config, metrics, root_folder="static_crop_50x50")


if __name__ == "__main__":
    train_model(
        #["The Eiffel Tower", "The Great Wall of China"],
        ["duck", "sheep"],
        "images",
        image_shape=(50, 50),
        num_epochs=10,
        batch_size=64,
        test_batch_size=128,
        lr=0.01,
        momentum=0.9,
        optimizer="Adam",
        cutmix_prob=0.8,
        resize_scale=(0.2, 1.0),
        logging_rate=1000,
        patience_length=3,
        patience_threshold=0.95,

        wandb_mode="online",
    )
