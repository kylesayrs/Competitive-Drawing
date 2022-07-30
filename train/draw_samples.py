import os
import cv2
import numpy

FILE_PATH = "../raw_data/duck.npy"
SAMPLE_DIR_PATH = "./"
NUM_SAMPLES = 10
IMAGE_SHAPE = (28, 28)

if __name__ == "__main__":
    all_images = numpy.load(FILE_PATH)

    for sample_i in range(NUM_SAMPLES):
        image = all_images[sample_i].reshape(IMAGE_SHAPE)
        out_path = os.path.join(SAMPLE_DIR_PATH, f"sample{sample_i}.png")
        cv2.imwrite(out_path, image)
