from typing import Optional

import cv2
import numpy
import argparse

from competitive_drawing import Settings
from competitive_drawing.train.utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument("root_dir")
parser.add_argument("out_path")
parser.add_argument("--filter", type=str, default=None)
parser.add_argument("--index", type=int, default=None)


def get_image_shape():
    image_size = Settings.get("IMAGE_SIZE", 50)
    return (image_size, image_size)


def get_class_names(filter_string: Optional[str]):
    if filter_string:
        return filter_string.split(",")
    else:
        return None
    

def get_image_index(index: int, num_images: int):
    if index is None:
        index = numpy.random.randint(0, num_images)

    if index >= num_images:
        raise ValueError(
            f"Passed --index={index}, but there are only {num_images} images "
            "in the dataset"
        )

    return index


if __name__ == "__main__":
    args = parser.parse_args()

    all_images, _, _ = load_data(
        args.root_dir, get_image_shape(), get_class_names(args.filter)
    )

    image_index = get_image_index(args.index, len(all_images))
    image_bitmap = all_images[image_index]
    image_bitmap.save(args.out_path)
    print(f"Successfully saved image to {args.out_path}")
