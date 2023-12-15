from typing import Tuple

import boto3
from competitive_drawing import Settings

S3_CLIENT = boto3.client("s3")


def get_uploaded_label_pairs():
    bucket = Settings().s3_models_bucket
    root_folder = Settings().s3_models_root_folder

    bucket_objects = S3_CLIENT.list_objects(Bucket=bucket)
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


def get_s3_dir(label_pair: Tuple[str, str]):
    label_pair_str = "-".join(sorted(list(label_pair)))

    return "/".join([Settings().s3_models_root_folder, label_pair_str])


def get_onnx_url(label_pair: Tuple[str, str]):
    s3_dir = get_s3_dir(label_pair)
    key = "/".join([s3_dir, "model.onnx"])

    response = S3_CLIENT.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": Settings().s3_models_bucket,
            "Key": key,
        },
        ExpiresIn=Settings().s3_model_duration
    )

    return response
