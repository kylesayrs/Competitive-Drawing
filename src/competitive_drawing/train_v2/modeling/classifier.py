import torch


class Classifier(torch.nn.Module):
    def __init__(self, num_classes: int = 2, temperature: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=False)

        def conv_block(in_ch: int, out_ch: int, dropout=0.0):
            layers = [
                torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                torch.nn.GroupNorm(1, out_ch),
                torch.nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(torch.nn.Dropout2d(dropout))
            return torch.nn.Sequential(*layers)

        # heavy inductive bias to maximize generalization
        self.conv = torch.nn.Sequential(
            *conv_block(1, 32),
            *conv_block(32, 32),
            *conv_block(32, 64),
            *conv_block(64, 128),
            *conv_block(128, 64),
        )

        self.fc = torch.nn.Linear(256, self.num_classes)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        # input.shape = (..., 28, 28)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        logits = self.fc(x)

        scores = self.softmax(logits / self.temperature)

        # logit norm to reduce overconfidence and increase generalization
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        logits = torch.div(logits, norms)
        
        #return logits, confs
        #torch.nn.functional.sigmoid(logits)
        return logits, scores
