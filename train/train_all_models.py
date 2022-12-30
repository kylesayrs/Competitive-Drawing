import os
import boto3
import random
import itertools

from utils import get_all_local_labels
from train_model import train_model


S3_CLIENT = boto3.client("s3")


def get_all_label_pairs(data_dir: str):
    label_names = get_all_local_labels(data_dir)
    pairs = itertools.permutations(label_names, 2)

    # remove pairs with incorrect ordering
    pairs = [
        [label_one, label_two]
        for label_one, label_two in pairs
        if label_one < label_two
    ]

    return pairs


def get_uploaded_label_pairs(root_folder: str):
    bucket_objects = S3_CLIENT.list_objects(Bucket="competitive-drawing-models-prod")
    if not "Contents" in bucket_objects:
        return []

    uploaded_label_pairs = []
    for object in bucket_objects["Contents"]:
        path_components = object["Key"].split("/")
        if path_components[0] == root_folder and path_components[-1] == "model.onnx":
            label_one, label_two = path_components[-2].split("-")
            if label_one < label_two:
                uploaded_label_pairs.append([label_one, label_two])
            else:
                print(f"WARNING: Invalid uploaded model {label_one}-{label_two}")

    return uploaded_label_pairs


def get_label_to_train(data_dir: str, root_folder: str):
    all_label_pairs = get_all_label_pairs(data_dir)
    uploaded_label_pairs = get_uploaded_label_pairs(root_folder)

    labels_pairs_to_train = [
        pair
        for pair in all_label_pairs
        if pair not in uploaded_label_pairs
    ]
    random.shuffle(labels_pairs_to_train)

    if len(labels_pairs_to_train) > 0:
        return labels_pairs_to_train[0]

    else:
        return None


if __name__ == "__main__":
    label_pair = get_label_to_train("images", "static_crop_50x50")
    while label_pair is not None:
        print(f"Training {label_pair}")
        train_model(
            label_pair,
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
        )

        label_pair = get_label_to_train("images", "static_crop_50x50")

    print("Done training all models")
