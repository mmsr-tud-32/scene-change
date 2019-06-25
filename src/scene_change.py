import argparse
import numpy as np
import cv2
from skimage.util import random_noise
from color_transfer import color_transfer


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


def match_colors(background, foreground):
    """
    Match the colours between the background and the foreground, maintaining the alpha channel
    of the foreground.

    :param background:
    :param foreground:
    :return:
    """
    transfered_foreground = color_transfer(background, foreground)
    return fix_alpha(foreground, transfered_foreground)


def fix_alpha(with_alpha, without_alpha):
    """
    Given two of the same image, one with an alpha channel and one without, add the alpha
    channel back into the image without, and return it.

    :param with_alpha:
    :param without_alpha:
    :return:
    """
    alpha_added = with_alpha.copy()
    alpha_added[:, :, :3] = without_alpha
    return alpha_added


def merge(foreground, background, offset=(0, 0)):
    """
    Given a background and a foreground, place the foreground
    on the background.

    :param offset:
    :param foreground:
    :param background:
    :return:
    """
    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = foreground[:, :, [3, 3, 3]].astype(float) / 255
    alpha = cv2.GaussianBlur(alpha, (11, 11), 0)

    foreground = foreground[:, :, :3]
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    cv2.imshow('foreground', foreground / 255)
    cv2.waitKey(0)
    cv2.imshow('background', background / 255)
    cv2.waitKey(0)
    merged = cv2.add(foreground, background) / 255

    # x_offset, y_offset = offset
    # merged = background.copy()
    # (height, width, _) = merged.shape
    # for y_pos, row in enumerate(foreground):
    #     if y_pos + y_offset >= height or y_pos + y_offset < 0:
    #         continue
    #
    #     for x_pos, pixel in enumerate(row):
    #         if x_pos + x_offset >= width or x_pos + x_offset < 0:
    #             continue
    #
    #         if pixel[3] == 0:
    #             continue
    #
    #         merged[y_pos + y_offset][x_pos + x_offset] = pixel[:3]

    return merged


if __name__ == "__main__":
    config = vars(get_arguments())
    scene_change(**config)
