import torch


class ClassEncoder(torch.nn.Module):
    def __init__(self, num_classes: int = 345, latent_size: int = 128):
        super(ClassEncoder, self).__init__()

        self.tokenizer = torch.nn.Linear(num_classes, latent_size)

    def forward(self, x: torch.tensor):
        return self.tokenizer(x)


class ImageEncoder(torch.nn.Module):
    def __init__(self, latent_size: int = 128):
        super(ImageEncoder, self).__init__()
        self.latent_size = latent_size

        self.conv = torch.nn.Sequential(
            *make_conv_block(1, 32, batch_normalization=False),
            *make_conv_block(32, 32),
            *make_conv_block(32, 64),
            *make_conv_block(64, 128),
            *make_conv_block(128, 64),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, self.latent_size)
        )


    def forward(self, x: torch.tensor):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        embedding =self.fc(x)

        return embedding


def make_conv_block(in_filters, out_filters, batch_normalization=True):
    block = [
        torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(0.2)
    ]
    
    if batch_normalization:
        block.append(torch.nn.BatchNorm2d(out_filters, 0.8))

    return block
