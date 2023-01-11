import os
import cv2
import tqdm
import numpy
import subprocess

SAMPLES_PER_CATEGORY = 1
TMP_DOWNLOAD_PATH = "./tmp.npy"
OUTPUT_PATH = "./samples/{category_name}.png"

def read_all_paths():
    all_paths = subprocess.Popen(
        "gsutil -m ls 'gs://quickdraw_dataset/full/numpy_bitmap'",
        shell=True,
        stdout=subprocess.PIPE
    ).stdout.read()
    all_paths = all_paths.decode("utf-8")
    all_paths = all_paths.split("\n")
    all_paths = all_paths[:-1]

    return all_paths

if __name__ == "__main__":
    all_paths = read_all_paths()

    for category_path in tqdm.tqdm(all_paths):
        category_name = category_path.split("/")[-1].split(".")[0]

        process = subprocess.Popen(f"gsutil -m cp '{category_path}' {TMP_DOWNLOAD_PATH}", shell=True)
        process.wait()

        all_images = numpy.load(TMP_DOWNLOAD_PATH)
        image = all_images[0].reshape((28, 28))
        cv2.imwrite(OUTPUT_PATH.format(category_name=category_name), image)

        os.remove(TMP_DOWNLOAD_PATH)
