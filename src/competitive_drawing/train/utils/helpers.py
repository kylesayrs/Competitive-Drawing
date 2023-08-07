from typing import Optional, List, Tuple

import os
import PIL
import tqdm
import numpy
from concurrent.futures import ThreadPoolExecutor


def get_all_local_labels(data_dir: str):
    label_names = []
    for file_name in os.listdir(data_dir):
        if file_name[0] == ".": continue

        label_name = file_name.split(".")[0]
        label_name = label_name.replace("-", " ")
        label_names.append(label_name)

    return label_names


def load_data(
    root_dir: str,
    image_shape: Tuple[int, int],
    class_names: Optional[List[str]] = None,
    one_hot: bool = False
):
    print("loading data...")
    class_names = class_names if class_names is not None else sorted(os.listdir(root_dir))
    class_images = [None for _ in range(len(class_names))]
    class_labels = [None for _ in range(len(class_names))]

    def load_class(class_index, class_name, progress):
        class_name = f"{class_name}.npy" if "npy" not in class_name else class_name
        file_path = os.path.join(root_dir, class_name)

        images = numpy.load(file_path)
        images = [PIL.Image.fromarray(array) for array in images]

        labels = [to_one_hot(class_index, len(class_names))] * len(images)

        class_images[class_index] = images
        class_labels[class_index] = labels
        progress.update(1)

    with ThreadPoolExecutor(max_workers=None) as executor:
        progress = tqdm.tqdm(total=len(class_names))
        futures = [
            executor.submit(load_class, class_index, class_name, progress)
            for class_index, class_name in enumerate(class_names)
        ]
        [future.result() for future in futures]

    all_images = sum(class_images, [])
    all_labels = sum(class_labels, [])

    print(f"loaded {len(all_images)} images and {len(all_labels)} labels")

    return all_images, all_labels, class_names

def to_one_hot(array, num_classes):
  array = numpy.array(array)
  one_hot = numpy.squeeze(numpy.eye(num_classes)[array.reshape(-1)])
  return numpy.array(one_hot, dtype=numpy.float32)
