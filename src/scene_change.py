import argparse
import numpy as np
import cv2
from skimage.util import random_noise


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('foreground')
    parser.add_argument('background')

    return parser.parse_args()


def scene_change(foreground, background):
    pass


def add_noise(image):
    noisy = (255 * random_noise(image, mode='gaussian', var=0.0007)).astype(np.uint8)
    return noisy


def add_blur(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / np.square(kernel_size)
    return cv2.filter2D(image, -1, kernel)


def merge(foreground, background):
    merged = background
    for row_idx, row in enumerate(foreground):
        for pixel_idx, pixel in enumerate(row):
            if pixel[3] == 0:
                continue
            merged[row_idx][pixel_idx] = pixel[:3]

    return merged


if __name__ == "__main__":
    config = vars(get_arguments())
    scene_change(**config)
