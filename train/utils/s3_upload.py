from typing import Dict

import json
import boto3
import torch

S3_CLIENT = boto3.client("s3")


def upload_model(model: torch.nn.Module, metrics: Dict[str, float]):
    # upload model onnx
    tmp_onnx_path = f"/tmp/model.onnx"
    tmp_metadata_path = f"/tmp/metadata.json"

    model = model.to("cpu")
    torch.onnx.export(
        model,
        torch.zeros([1, 1] + list(wandb.config["image_shape"])),
        tmp_onnx_path,
        input_names=["input"],
        output_names=["logits", "output"],
        opset_version=14
    )
    model = model.to(DEVICE)

    S3_CLIENT.upload_file(
        tmp_onnx_path,
        "competitive-drawing-models-prod",
        f"{wandb.config['model_name']}/model.onnx"
    )

    # upload associated config file
    metadata = dict(wandb.config)
    metadata.update(metrics)
    with open(tmp_metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file)

    S3_CLIENT.upload_file(
        tmp_metadata_path,
        "competitive-drawing-models-prod",
        f"{wandb.config['model_name']}/metadata.json"
    )

    # remove temporary files
    os.remove(tmp_onnx_path)
    os.remove(tmp_metadata_path)

    print("Uploaded model files")
