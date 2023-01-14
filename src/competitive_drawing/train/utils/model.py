import torch


class Classifier(torch.nn.Module):
    def __init__(self, num_classes: int = 2, temperature: float = 0.05):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.temperature = torch.nn.Parameter(torch.tensor(temperature, dtype=torch.float32), requires_grad=False)

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

        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        logits_normed = torch.div(logits, norms) / self.temperature
        scores = self.softmax(logits_normed)

        #return logits, confs
        return logits_normed, scores
