import torch


class Critic(torch.nn.module):
    def __init__(self, logit_norm: bool = False):
        super().__init__()
        self.logit_norm = logit_norm

        self.conv = torch.nn.Sequential(
            self.make_conv_block(1, 32, bn=False)
            *make_conv_block(32, 32),
            *make_conv_block(32, 64),
            *make_conv_block(64, 128),
            *make_conv_block(128, 64),
        )

        self.linear = torch.nn.Linear(256, 4)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, image: torch.tensor):
        x = self.conv(image)
        logits = self.linear(x)

        if self.logit_norm:
            norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
            logits_normed = torch.div(logits, norms) / self.temperature
            logits = self.softmax(logits_normed)

        return self.softmax(logits)


def make_conv_block(in_filters: int, out_filters: int, batch_norm: bool =True):
        block = [
            torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.2)
        ]
        if batch_norm:
            block.append(torch.nn.BatchNorm2d(out_filters, 0.8))
