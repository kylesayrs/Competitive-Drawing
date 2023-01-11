import torch
import pickle

from .s3 import get_object_file_stream
from drawnt import Settings


class Classifier(torch.nn.Module):
    def __init__(self, num_classes: int = 2):
        super(Classifier, self).__init__()
        self.num_classes = num_classes

        def conv_block(in_filters, out_filters, bn=True):
            block = [torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1), torch.nn.ReLU(), torch.nn.Dropout2d(0.2)]
            if bn:
                block.append(torch.nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv = torch.nn.Sequential(
            *conv_block(1, 32, bn=False),
            *conv_block(32, 32),
            *conv_block(32, 64),
            *conv_block(64, 128),
            *conv_block(128, 64),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, self.num_classes),
        )

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        logits = self.fc(x)
        confs = self.softmax(logits)

        return logits, confs


def get_model_class():
    """
    I can't figure out how to get pickling the original class to work
    so this is my lazy solution
    """
    return Classifier


    bucket = Settings.get("S3_MODELS_BUCKET", "competitive-drawing-models-prod")

    root_folder = Settings.get("S3_MODELS_ROOT_FOLDER", "static_crop_50x50")
    key = f"{root_folder}/model.pkl"

    pickled_file_stream = get_object_file_stream(bucket, key)
    return pickle.loads(torch.load(pickled_file_stream))
