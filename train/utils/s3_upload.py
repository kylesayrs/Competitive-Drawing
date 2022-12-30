from typing import Dict

import os
import json
import boto3
import torch

S3_CLIENT = boto3.client("s3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def upload_model(
    model: torch.nn.Module,
    wandb_config: Dict[str, any],
    metrics: Dict[str, float],
    root_folder: str = ""
):
    # upload model onnx
    tmp_onnx_path = f"/tmp/model.onnx"
    tmp_metadata_path = f"/tmp/metadata.json"
    tmp_torch_path = f"/tmp/model.pth"

    model = model.to("cpu")
    torch.onnx.export(
        model,
        torch.zeros([1, 1] + list(wandb_config["image_shape"])),
        tmp_onnx_path,
        input_names=["input"],
        output_names=["logits", "output"],
        opset_version=14
    )
    model = model.to(DEVICE)

    S3_CLIENT.upload_file(
        tmp_onnx_path,
        "competitive-drawing-models-prod",
        f"{root_folder}/{wandb_config['model_name']}/model.onnx"
    )

    # upload associated config file
    metadata = dict(wandb_config)
    metadata.update(metrics)
    with open(tmp_metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file)

    S3_CLIENT.upload_file(
        tmp_metadata_path,
        "competitive-drawing-models-prod",
        f"{root_folder}/{wandb_config['model_name']}/metadata.json"
    )

    # upload torch weights
    torch.save(model.state_dict(), tmp_torch_path)

    S3_CLIENT.upload_file(
        tmp_torch_path,
        "competitive-drawing-models-prod",
        f"{root_folder}/{wandb_config['model_name']}/model.pth"
    )

    # remove temporary files
    os.remove(tmp_onnx_path)
    os.remove(tmp_metadata_path)
    os.remove(tmp_torch_path)

    print("Uploaded model files")
