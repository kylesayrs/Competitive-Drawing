from typing import Optional, List, Tuple

import os
import PIL
import numpy


def get_all_local_labels(data_dir: str):
    label_names = []
    for file_name in os.listdir(data_dir):
        if file_name[0] == ".": continue

        label_name = file_name.split(".")[0]
        label_name.replace("-", " ")
        label_names.append(label_name)

    return label_names


def load_data(
    root_dir: str,
    image_shape: Tuple[int, int],
    class_names: Optional[List[str]] = None
):
    all_images = []
    all_labels = []
    label_names = []
    class_names = class_names if class_names is not None else os.listdir(root_dir)
    for file_i, file_name in enumerate(class_names):
        file_name += ".npy"
        file_path = os.path.join(root_dir, file_name)
        images = numpy.load(file_path)
        images = images.reshape(-1, *image_shape)
        images = [PIL.Image.fromarray(array) for array in images]
        all_images.extend(images)
        all_labels.extend([file_i] * len(images))
        label_names.append(os.path.splitext(os.path.basename(file_name))[0])

    print(f"loaded {len(all_images)} images, {len(all_labels)} labels, from {label_names}")

    return all_images, all_labels, label_names
