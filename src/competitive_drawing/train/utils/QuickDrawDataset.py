import torch
from torchvision import transforms

from competitive_drawing.train.utils.RandomResizePad import RandomResizePad


class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, is_test: bool = False):
        self.X = X
        self.y = y
        if not is_test:
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(5, shear=5),
                transforms.ToTensor(),
                RandomResizePad(scale=(0.3, 1.0), value=0)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if not self.transform:
            return image, label
        else:
            return self.transform(image), label
