import torch
import numpy


class ClassEncoder(torch.nn.Module):
    def __init__(self, num_classes: int = 345, latent_size: int = 128):
        super(ClassEncoder, self).__init__()

        self.tokenizer = torch.nn.Linear(num_classes, latent_size)

    def forward(self, x: torch.tensor):
        x = self.tokenizer(x)
        return torch.nn.functional.normalize(x, p=2)


class ImageEncoder(torch.nn.Module):
    """
    50x50 image size
    """
    def __init__(self, latent_size: int = 128, max_temp: float = 0.0):
        super(ImageEncoder, self).__init__()
        self.latent_size = latent_size
        self.max_temp = max_temp

        if max_temp < 0.0:
            raise ValueError("max_temp must be greater than 0.0")

        self.conv = torch.nn.Sequential(
            *make_conv_block(1, 64, batch_normalization=False),
            *make_conv_block(64, 128),
            *make_conv_block(128, 128),
            *make_conv_block(128, 128),
            *make_conv_block(128, 64),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, self.latent_size)
        )

        self.temperature = torch.nn.Parameter(torch.tensor(min(0.07, self.max_temp)))


    def forward(self, x: torch.tensor):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
    
        x = torch.nn.functional.normalize(x, p=2)
    
        if self.training and self.temperature > 0.0:
            x = x * torch.exp(torch.clip(self.temperature, None, self.max_temp))

        return x


def make_conv_block(in_filters, out_filters, batch_normalization=True):
    block = [
        torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.2)
    ]
    
    if batch_normalization:
        block.append(torch.nn.BatchNorm2d(out_filters, 0.8))

    return block


def get_num_trainable_parameters(model: torch.nn.Module):
    model_parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
    return sum([numpy.prod(parameter.size()) for parameter in model_parameters])


if __name__ == "__main__":
    num_classes = 10
    class_encoder = ClassEncoder(num_classes)
    image_encoder = ImageEncoder()

    class_input = torch.zeros((1, num_classes, ), dtype=torch.float32)
    class_encoder(class_input)
    print(get_num_trainable_parameters(class_encoder))

    image_input = torch.zeros((1, 1, 50, 50), dtype=torch.float32)
    image_encoder(image_input)
    print(get_num_trainable_parameters(image_encoder))
