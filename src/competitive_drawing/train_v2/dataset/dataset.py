import json
import torch
from torchvision import transforms

from competitive_drawing.train.utils.RandomResizePad import RandomResizePad
from competitive_drawing.train_v2.dataset.VecToRaster import VecToRaster


class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *file_paths: list[str],
        output_size: tuple[int, int] = (64, 64),
        use_unrecognized: bool = False,
        is_test: bool = False
    ):
        self.images = []
        self.labels = []

        for index, file_path in enumerate(file_paths):
            images = load_vectors_from_file(file_path, use_unrecognized=use_unrecognized)
            self.images.extend(images)
            self.labels.extend([index] * len(images))

        self.vec_to_raster = VecToRaster(side=28, line_diameter=16, padding=16)
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(5, shear=5),
            RandomResizePad(scale=(0.3, 1.0), value=0),
            transforms.Resize(output_size),
        ])
        self.test_transform = transforms.Resize(output_size)
        self.is_test = is_test

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def __getitem__(self, idx: int):
        # get image
        image = self.images[idx]
        image = self.vec_to_raster(image)
        image = image.unsqueeze(0)  # add batch dim
        if self.is_test:
            image = self.test_transform(image)
        else:
            image = self.train_transform(image)

        # get label
        label = self.labels[idx]

        return image, label


def load_vectors_from_file(file_path: str, use_unrecognized: bool) -> list[list[list[int]]]:
    vector_images = []
    with open(file_path, "r") as stroke_file:
        for line in stroke_file:
            drawing_data = json.loads(line)
            if drawing_data["recognized"] or use_unrecognized:
                vector_images.append(drawing_data["drawing"])

    return vector_images


if __name__ == "__main__":
    from torchvision.io import write_png
    from torch.utils.data import DataLoader, RandomSampler

    dataset = QuickDrawDataset(
        "src/competitive_drawing/train_v2/camera.ndjson",
        "src/competitive_drawing/train_v2/coffee_cup.ndjson",
        is_test=False,
    )

    data_loader = DataLoader(dataset, batch_size=1, sampler=RandomSampler(dataset))
    
    image, label = next(iter(data_loader))

    print(label[0])
    write_png(image[0], f"sample.png")