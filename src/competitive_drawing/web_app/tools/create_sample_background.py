import os
import cv2
import numpy
import random

SAMPLES_DIR_PATH = "./samples"
NUM_ROWS = 18
NUM_COLUMNS = 18
BORDER = 30
RAND_PLACEMENT = False
OUTFILE_PATH = "background.png"
SEED = 42

STROKE_COLOR = [0, 0, 0]
BACKGROUND_COLOR = [229, 245, 251]

if __name__ == "__main__":
    # seed randomness
    random.seed(SEED)

    # read images
    images = []
    total_num_samples = NUM_ROWS * NUM_COLUMNS
    file_names = [
        file_name
        for file_name in os.listdir(SAMPLES_DIR_PATH)
        if file_name[0] != "."
    ]
    assert total_num_samples <= len(file_names), "Not enough samples"
    for file_name in file_names[:total_num_samples]:
        file_path = os.path.join(SAMPLES_DIR_PATH, file_name)

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)

    # calculate sizes
    random.shuffle(images)
    images = numpy.array(images)
    image_shape = images.shape[1:3]
    image_shape_with_border = [
        image_shape[0] + 2 * BORDER,
        image_shape[1] + 2 * BORDER
    ]

    # create background
    width = NUM_ROWS * image_shape_with_border[1]
    height = NUM_COLUMNS * image_shape_with_border[0]
    background = numpy.full((height, width, 3), BACKGROUND_COLOR)

    # place images
    for image_index, image in enumerate(images):
        image_bgr = numpy.array([
            [
                (
                    numpy.array(STROKE_COLOR) * (image[y, x] / 255) +
                    numpy.array(BACKGROUND_COLOR) * ((255 - image[y, x]) / 255)
                )
                for x in range(image.shape[1])
            ]
            for y in range(image.shape[0])
        ], dtype=numpy.int32)

        random_x = random.randint(0, BORDER * 2) if RAND_PLACEMENT else BORDER
        random_y = random.randint(0, BORDER * 2) if RAND_PLACEMENT else BORDER
        image_with_border = cv2.copyMakeBorder(
            image_bgr,
            random_y, BORDER * 2 - random_y, random_x, BORDER * 2 - random_x,
            cv2.BORDER_CONSTANT, value=BACKGROUND_COLOR
        )

        x_position = image_index % NUM_ROWS
        y_position = image_index // NUM_ROWS
        background[
            y_position * image_shape_with_border[0]: (y_position + 1) * image_shape_with_border[0],
            x_position * image_shape_with_border[1]: (x_position + 1) * image_shape_with_border[1],
        ] = image_with_border

    # save border file
    cv2.imwrite(OUTFILE_PATH, background)
