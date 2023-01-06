import os
import cv2
import numpy

FILE_PATH = "data/The Eiffel Tower.npy"
IMAGE_SHAPE = (50, 50)
SAVE_PATH = "./sample.png"

if __name__ == "__main__":
    all_images = numpy.load(FILE_PATH)

    image = all_images[0]
    image = image.reshape(IMAGE_SHAPE)
    if SAVE_PATH is not None:
        cv2.imwrite(SAVE_PATH, image)
    else:
        cv2.imshow("sample", image)
        cv2.waitKey(0)
