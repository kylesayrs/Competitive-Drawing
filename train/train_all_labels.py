import os
import boto3
import random
import itertools

from helpers import get_all_local_labels
from train_model import train_model


S3_CLIENT = boto3.client("s3")


if __name__ == "__main__":
    all_local_labels = get_all_local_labels()

    print(f"Training {all_local_labels}")

    train_model(
        all_local_labels,
        num_epochs=50,
        batch_size=256,
        test_batch_size=256,
        lr=0.01,
        momentum=None,
        optimizer="Adam",
        cutmix_prob=0.0,
        logging_rate=1,
        patience_length=None,
        patience_threshold=None,
        model_name="Megamodel",
        wandb_mode="disabled",
    )

    print("Done training all models")
