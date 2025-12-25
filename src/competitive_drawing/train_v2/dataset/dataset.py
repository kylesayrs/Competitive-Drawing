import os
import json
import torch
from torchvision import transforms

from competitive_drawing.train_v2.dataset.VecToRaster import VecToRaster
from competitive_drawing.train_v2.utils import smoothed_one_hot


class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images: list,
        labels: list[int],
        side: int = 28,
        line_diameter: float = (1 / 32),
        padding: int = (1 / 32),
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        augment: bool = True
    ):
        self.augment = augment
        self.images = images
        self.labels = labels
        self.num_classes = max(labels) + 1

        self.vec_to_raster = VecToRaster(
            side=side,
            line_diameter=line_diameter,
            padding=padding,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        self.augmentations = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(5, shear=5),
        ])

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, int]:
        # get image
        image = self.images[idx]
        image, partial_frac = self.vec_to_raster(image, augment=self.augment)
        image = image.unsqueeze(0)  # add batch dim
        if self.augment:
            image = self.augmentations(image)

        # get label
        label = torch.zeros((self.num_classes, ))
        label[self.labels[idx]] = partial_frac

        return image, label
    

def load_data(data_dir: str, class_names: list[str], use_unrecognized: bool = False) -> tuple[list, list[int]]:
    images = []
    labels = []

    for index, class_name in enumerate(class_names):
        file_path = os.path.join(data_dir, f"{class_name}.ndjson")
        _images = load_vectors_from_file(file_path, use_unrecognized=use_unrecognized)
        images.extend(_images)
        labels.extend([index] * len(_images))

    assert len(images) == len(labels)

    return images, labels


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

    images, labels = load_data("src/competitive_drawing/train_v2", ["camera", "coffee_cup"])

    dataset = QuickDrawDataset(images, labels, augment=True)

    image, label = dataset[8]
    image = (image * 255).to(torch.uint8)
    print(label)
    write_png(image, f"sample.png")