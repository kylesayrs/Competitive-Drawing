from typing import List, Optional

import os
import torch
import wandb

from sklearn.model_selection import train_test_split

from competitive_drawing.train_v2.dataset.dataset import QuickDrawDataset, load_data
from competitive_drawing.train_v2.modeling.classifier import Classifier
from competitive_drawing.train_v2.train.utils import accuracy_score
from competitive_drawing.train_v2.utils import collect_func_args


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


def train_model(
    # data
    class_names: List[str],
    data_dir: str,
    use_unrecognized: bool = False,
    image_size: int = 64,
    # data argumentations
    min_scale: float = 0.5,
    max_scale: float = 1.0,
    # model
    dtype: torch.dtype = torch.bfloat16,
    # training
    num_epochs: int = 6,
    batch_size: int = 128,
    test_batch_size: int = 128,
    test_size: float = 0.2,
    lr: float = 0.005,
    optimizer: str = "SGD",
    momentum: float = 0.9,
    # productionized training
    model_name: Optional[str] = None,
    patience_length: Optional[int] = 3,
    patience_threshold: Optional[float] = 0.95,
    logging_rate: int = 10,
    save_checkpoints: bool = False,
    do_upload: bool = True,
    wandb_mode: str = "online",
):
    args = collect_func_args()
    assert class_names[0] < class_names[1], "class names are out of order!"

    # wandb
    model_name = model_name or "-".join(class_names)
    run = wandb.init(
        project="competitive_drawing_classifier",
        entity="kylesayrs",
        name=model_name,
        reinit=True,
        mode=wandb_mode,
        config=args,
    )
    print(wandb.config)

    # load data
    images, labels = load_data(data_dir, class_names, use_unrecognized=use_unrecognized)

    # create datasets
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, shuffle=True)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    # create data loaders
    augmentations = dict(side=image_size, min_scale=min_scale, max_scale=max_scale)
    train_dataset = QuickDrawDataset(x_train, y_train, **augmentations, augment=True)
    test_dataset = QuickDrawDataset(x_test, y_test, **augmentations, augment=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                              shuffle=True, num_workers=0, drop_last=True)

    # set up model, optimizer, and criterion
    with DEVICE:
        model = Classifier(len(class_names)).to(dtype)
        if optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer {optimizer}")
        criterion = torch.nn.MSELoss().to(device=DEVICE, dtype=dtype)

    # train
    epoch_test_accs = []
    test_accuracy = 0.0
    metrics = {}

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device=DEVICE, dtype=dtype)
            labels = labels.to(device=DEVICE, dtype=dtype)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits, scores = model(images)
            print(scores[0])
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % logging_rate == logging_rate - 1:
                train_accuracy = accuracy_score(
                    torch.argmax(labels, axis=-1), torch.argmax(scores, axis=-1))

                with torch.no_grad():
                    test_images, test_labels = next(iter(test_loader))
                    test_images = test_images.to(device=DEVICE, dtype=dtype)
                    test_labels = test_labels.to(device=DEVICE, dtype=dtype)
                    test_logits, test_scores = model(test_images)
                    test_accuracy = accuracy_score(
                        torch.argmax(test_labels, axis=-1), torch.argmax(test_scores, axis=-1))
                    test_loss = criterion(test_logits, test_labels)

                print(
                    f"[{epoch + 1}, {i + 1:5d}] "
                    f"loss: {running_loss / logging_rate} "
                    f"test_acc: {test_accuracy}"
                )
                metrics = {
                    "loss": running_loss / logging_rate,
                    "test_loss": test_loss.item(),
                    "train_acc": train_accuracy,
                    "test_acc": test_accuracy,
                }
                wandb.log(metrics)

                running_loss = 0.0

        # check patience each epoch
        epoch_test_accs.append(test_accuracy)
        if (
            patience_length and patience_threshold and
            len(epoch_test_accs) >= patience_length and
            all([acc >= patience_threshold for acc in epoch_test_accs[-1 * patience_length:]])
        ):
            break

        # save each epoch
        if save_checkpoints:
            os.makedirs(f"./checkpoints/{run.id}", exist_ok=True)
            print(f"saving model to ./checkpoints/{run.id}/epoch{epoch}.pth")
            torch.save(model.state_dict(), f"./checkpoints/{run.id}/epoch{epoch}.pth")

    if do_upload:
        pass
        #upload_model(model, wandb.config, metrics, root_folder="static_crop_50x50")


if __name__ == "__main__":
    train_model(["camera", "coffee_cup"], "src/competitive_drawing/train_v2", wandb_mode="offline")
