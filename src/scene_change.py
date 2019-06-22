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


def position(foreground, fg_vp, background, bg_vp):
    flip = True
    x = 1
    y = 2
    scale = 1

    (fg_h, fg_w) = foreground.shape[:2]
    (bg_h, bg_w) = background.shape[:2]

    assert fg_h <= bg_h
    assert fg_w <= bg_w

    (fg_vp_x, fg_vp_y) = fg_vp
    (bg_vp_x, bg_vp_y) = bg_vp

    # normal case no smart thing to do
    scale = 1
    flip = False

    x = (bg_w - fg_w) / 2
    y = bg_h - fg_h

    # normalize vanishing points
    # the foreground vanishing point expressed in the coordinates of the background

    fg_vp_x_n = fg_vp_x + x
    fg_vp_y_n = fg_vp_y + y

    if fg_vp_y_n < bg_vp_y:
        y = y + (bg_vp_y - fg_vp_y_n)

    x = x + (bg_vp_x - fg_vp_x_n)

    return [x, y], scale, flip


def add_noise(image):
    """
    Add random noise to an image.

    :param image:
    :return:
    """
    noisy = (255 * random_noise(image, mode='gaussian', var=0.0007)).astype(np.uint8)
    return noisy


def add_blur(image, kernel_size=3):
    """
    Blur an image.

    :param image:
    :param kernel_size:
    :return:
    """
    kernel = np.ones((kernel_size, kernel_size), np.float32) / np.square(kernel_size)
    return cv2.filter2D(image, -1, kernel)


def mirror(image):
    """
    Mirror image.

    :param image:
    :return:
    """
    return cv2.flip(image, 1)


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
