import argparse
import numpy as np
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


if __name__ == "__main__":
    config = vars(get_arguments())
    scene_change(**config)
