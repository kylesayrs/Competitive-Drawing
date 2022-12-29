import torch


class Classifier(torch.nn.Module):
    def __init__(self, num_classes: int):
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
