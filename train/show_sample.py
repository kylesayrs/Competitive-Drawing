import os
import cv2
import numpy

FILE_PATH = "data/The Eiffel Tower.npy"
IMAGE_SHAPE = (50, 50)

if __name__ == "__main__":
    all_images = numpy.load(FILE_PATH)

    image = all_images[0]
    image = image.reshape(IMAGE_SHAPE)
    cv2.imshow("sample", image)
    cv2.waitKey(0)
