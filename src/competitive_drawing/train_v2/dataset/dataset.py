import json
import torch
from torchvision import transforms

from competitive_drawing.train_v2.dataset.VecToRaster import VecToRaster


class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *file_paths: list[str],
        side: int = 28,
        line_diameter: float = (1 / 16),
        padding: int = (1 / 16),
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        use_unrecognized: bool = False,
        train: bool = True
    ):
        self.images = []
        self.labels = []
        self.train = train

        for index, file_path in enumerate(file_paths):
            images = load_vectors_from_file(file_path, use_unrecognized=use_unrecognized)
            self.images.extend(images)
            self.labels.extend([index] * len(images))

        self.vec_to_raster = VecToRaster(
            side=side,
            line_diameter=line_diameter,
            padding=padding,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(5, shear=5),
        ])

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def __getitem__(self, idx: int):
        # get image
        image = self.images[idx]
        image = self.vec_to_raster(image)
        image = image.unsqueeze(0)  # add batch dim
        if self.train:
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

    dataset = QuickDrawDataset(
        "src/competitive_drawing/train_v2/camera.ndjson",
        "src/competitive_drawing/train_v2/coffee_cup.ndjson",
        train=True,
    )

    image, label = dataset[8]
    print(label)
    write_png(image, f"sample.png")