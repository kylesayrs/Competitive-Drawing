from typing import Tuple

import boto3
from competitive_drawing import Settings

S3_CLIENT = boto3.client("s3")


def get_uploaded_label_pairs():
    bucket = Settings.get("S3_MODELS_BUCKET", "competitive-drawing-models-prod")
    root_folder = Settings.get("S3_MODELS_ROOT_FOLDER", "static_crop_50x50")

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
    root_folder = Settings.get("S3_MODELS_ROOT_FOLDER", "static_crop_50x50")

    label_pair_str = "-".join(sorted(list(label_pair)))

    return "/".join([root_folder, label_pair_str])


def get_onnx_url(label_pair: Tuple[str, str]):
    s3_dir = get_s3_dir(label_pair)
    key = "/".join([s3_dir, "model.onnx"])

    response = S3_CLIENT.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": Settings.get("S3_MODELS_BUCKET", "competitive-drawing-models-prod"),
            "Key": key,
        },
        ExpiresIn=Settings.get("S3_MODEL_DURATION", 108000)  # 30 minutes
    )

    return response
