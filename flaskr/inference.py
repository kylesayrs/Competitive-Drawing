import os
import torch
from torchvision.transforms.functional import to_tensor

from train.train_model import Classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = Classifier()

    model_checkpoint_path = os.environ.get("MODEL_PATH", "./flaskr/static/models/model.pth")
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE))

    model = model.eval()
    return model

# TODO inference class
def infer_image(model, image):
    alpha_channel = image.split()[-1]
    input = to_tensor(alpha_channel)
    input = torch.reshape(input, (1, 1, 28, 28))
    with torch.no_grad():
        logits, confidences = model(input)
    return logits[0].tolist()
