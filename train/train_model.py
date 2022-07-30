import os
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

# needed to define model for exporting to onnx
NUM_CLASSES = 10

# load data
def load_data(root_dir, class_names=None):
    all_images = []
    all_labels = []
    label_names = []
    class_names = class_names if class_names is not None else os.listdir(root_dir)
    assert len(class_names) == NUM_CLASSES == wandb.config["num_classes"]
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
    def __init__(self):
        super(Classifier, self).__init__()

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
            nn.Linear(128, NUM_CLASSES),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        logits = self.fc(x)
        confs = self.softmax(logits)

        return logits, confs

if __name__ == "__main__":
    # config
    run = wandb.init(
        project="competitive_drawing",
        entity="kylesayrs",
        mode="online"
    )
    wandb.config = {
        "image_shape": (28, 28),
        "num_epochs": 30,
        "batch_size": 256,
        "num_classes": NUM_CLASSES,
        "lr": 0.0001,
        "logging_rate": 1000,
        "test_batch_size": 256,
        "mixup_alpha": 0.3,
        "cutmix_alpha": 0.0,
        "cutmix_prob": 1.0,
        "label_smoothing": 0.1,
    }
    assert wandb.config["num_classes"] == NUM_CLASSES

    CMAP = "gray_r"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    all_images, all_labels, label_names = load_data(
        "raw_data",
        class_names=[
            "sheep",
            "guitar",
            "pig",
            "tree",
            "clock",
            "squirrel",
            "duck",
            "panda",
            "spider",
            "snowflake",
        ]
    )

    # create datasets
    x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.20, shuffle=True, random_state=42)
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

                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / wandb.config["logging_rate"]} test_acc: {test_accuracy}')
                wandb.log({
                    "loss": running_loss / wandb.config["logging_rate"],
                    "test_loss": test_loss,
                    "train_acc": train_accuracy,
                    "test_acc": test_accuracy,
                })

                running_loss = 0.0

        # save each epoch
        if run:
            os.makedirs(f"./checkpoints/{run.id}", exist_ok=True)
            print(f"saving model to ./checkpoints/{run.id}/epoch{epoch}.pth")
            torch.save(model.state_dict(), f"./checkpoints/{run.id}/epoch{epoch}.pth")
        else:
            print(f"saving model to ./checkpoints/epoch{epoch}.pth")
            torch.save(model.state_dict(), f"./checkpoints/epoch{epoch}.pth")

    print('Finished Training')
