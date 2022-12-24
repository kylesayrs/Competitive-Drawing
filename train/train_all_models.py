import os
import boto3
import random
import itertools

from helpers import get_all_local_labels
from train_model import train_model


S3_CLIENT = boto3.client("s3")


def get_all_label_pairs():
    label_names = get_all_local_labels()
    pairs = itertools.permutations(label_names, 2)

    # remove pairs with incorrect ordering
    pairs = [
        [label_one, label_two]
        for label_one, label_two in pairs
        if label_one < label_two
    ]

    return pairs


def get_uploaded_label_pairs():
    bucket_objects = S3_CLIENT.list_objects(Bucket="competitive-drawing-models-prod")
    if not "Contents" in bucket_objects:
        return []

    uploaded_label_pairs = []
    for object in bucket_objects["Contents"]:
        if object["Key"].split("/")[-1] == "model.onnx":
            label_one, label_two = object["Key"].split("/")[0].split("-")
            if label_one < label_two:
                uploaded_label_pairs.append([label_one, label_two])
            else:
                print(f"WARNING: Invalid uploaded model {label_one}-{label_two}")

    return uploaded_label_pairs


if __name__ == "__main__":
    all_label_pairs = get_all_label_pairs()
    uploaded_label_pairs = get_uploaded_label_pairs()

    labels_pairs_to_train = [
        pair
        for pair in all_label_pairs
        if pair not in uploaded_label_pairs
    ]
    random.shuffle(labels_pairs_to_train)


    for label_pair in labels_pairs_to_train:
        print(f"Training {label_pair}")
        train_model(
            label_pair,
            num_epochs=10,
            batch_size=128,
            test_batch_size=128,
            lr=0.008,
            momentum=0.9,
            optimizer="SGD",
            cutmix_prob=0.8,
            logging_rate=500,
            patience_length=3,
            patience_threshold=0.95,
        )

    print("Done training all models")
