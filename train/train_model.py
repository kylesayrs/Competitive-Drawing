from typing import List, Optional, Tuple

import os
import json
import numpy
import matplotlib.pyplot as plt
import PIL
import cv2
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torchvision import transforms
from timm.data import Mixup


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# load data
def load_data(root_dir: str, class_names: Optional[List[str]]=None):
    all_images = []
    all_labels = []
    label_names = []
    class_names = class_names if class_names is not None else os.listdir(root_dir)
    for file_i, file_name in enumerate(class_names):
        file_name += ".npy"
        file_path = os.path.join(root_dir, file_name)
        images = numpy.load(file_path)
        images = images.reshape(-1, *wandb.config["image_shape"])
        images = [PIL.Image.fromarray(array) for array in images]
        all_images.extend(images)
        all_labels.extend([file_i] * len(images))
        label_names.append(os.path.splitext(os.path.basename(file_name))[0])

    print(f"loaded {len(all_images)} images, {len(all_labels)} labels, from {label_names}")

    return all_images, all_labels, label_names


# dataset
class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

        assert len(self.X) == len(self.y)

    def add_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if not self.transform:
            return image, label
        else:
            return self.transform(image), label


# model
class Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super(Classifier, self).__init__()
        self.num_classes = num_classes

        def conv_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.ReLU(), nn.Dropout2d(0.2)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv = nn.Sequential(
            *conv_block(1, 16, bn=False),
            *conv_block(16, 32),
            *conv_block(32, 64),
            *conv_block(64, 64),
            *conv_block(64, 128),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, self.num_classes),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        logits = self.fc(x)
        confs = self.softmax(logits)

        return logits, confs


def save_model(model: nn.Module):
    # save model onnx
    torch.onnx.export(
        model,
        torch.zeros([1, 1] + list(wandb.config.image_shape)),
        os.path.join(wandb.config["save_dir"], "model.onnx"),
        input_names=["input"],
        output_names=["logits", "output"],
        opset_version=14
    )

    # save associated config file
    with open(os.path.join(wandb.config["save_dir"], "config.json"), "w") as config_file:
        json.dump(config_file, dict(wandb.config))


def train_model(
    class_names: List[str],
    save_dir: str,
    num_epochs: int = 30,
    batch_size: int = 256,
    test_batch_size: int = 256,
    test_size: float = 0.2,
    lr: float = 0.0001,
    image_shape: Tuple[int, int] = (28, 28),
    mixup_alpha: float = 0.3,
    cutmix_alpha: float = 0.0,
    cutmix_prob: float = 1.0,
    label_smoothing: float = 0.1,
    logging_rate: int = 1000,
    save_checkpoints: bool = False,
):
    # config
    run = wandb.init(
        project="competitive_drawing",
        entity="kylesayrs",
        name="_".join(class_names),
        reinit=True,
        mode="online",
    )
    wandb.config = {
        "image_shape": image_shape,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "num_classes": len(class_names),
        "lr": lr,
        "logging_rate": logging_rate,
        "test_batch_size": test_batch_size,
        "test_size": test_size,
        "mixup_alpha": mixup_alpha,
        "cutmix_alpha": cutmix_alpha,
        "cutmix_prob": cutmix_prob,
        "label_smoothing": label_smoothing,
        "save_dir": save_dir,
        "save_checkpoints": save_checkpoints,
        "run_id": run.id,
    }

    # transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(5, shear=5),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_mixup = Mixup(
        mixup_alpha=wandb.config["mixup_alpha"],
        cutmix_alpha=wandb.config["cutmix_alpha"],
        prob=wandb.config["cutmix_prob"],
        label_smoothing=wandb.config["label_smoothing"],
        num_classes=wandb.config["num_classes"]
    )

    # load data
    all_images, all_labels, label_names = load_data("raw_data", class_names=class_names)

    # create datasets
    x_train, x_test, y_train, y_test = train_test_split(
        all_images,
        all_labels,
        test_size=wandb.config["test_size"],
        shuffle=True,
        random_state=42
    )
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    train_dataset = QuickDrawDataset(x_train, y_train, transform=train_transform)
    test_dataset = QuickDrawDataset(x_test, y_test, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config["batch_size"],
                                              shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=wandb.config["test_batch_size"],
                                              shuffle=True, num_workers=2)

    model = Classifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["lr"])
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    print(model)

    # train
    print("begin training")
    for epoch in range(wandb.config["num_epochs"]):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (images, raw_labels) in enumerate(train_loader):
            # mixup/ cutmix
            images, cutmix_labels = train_mixup(images, raw_labels)

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
                wandb.log({
                    "loss": running_loss / wandb.config["logging_rate"],
                    "test_loss": test_loss,
                    "train_acc": train_accuracy,
                    "test_acc": test_accuracy,
                })

                running_loss = 0.0

        # save each epoch
        if wandb.config["save_checkpoints"]:
            os.makedirs(f"./checkpoints/{run.id}", exist_ok=True)
            print(f"saving model to ./checkpoints/{run.id}/epoch{epoch}.pth")
            torch.save(model.state_dict(), f"./checkpoints/{run.id}/epoch{epoch}.pth")

    print("Finished Training")
    save_model(model)


if __name__ == "__main__":
    train_model(["sheep", "clock"], save_path="models")
