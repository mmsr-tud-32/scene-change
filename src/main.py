import argparse
import random
from glob import glob

import cv2

from scene_change import merge
from vanishing_point import resize_image


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('fg_path', help='Path to folder containing foreground')
    parser.add_argument('bg_path', help='Path to folder containing background')

    return parser.parse_args()


def composite_images(fg_path, bg_path):
    while True:
        fg_file = random.choice(glob(fg_path + '/*'))
        bg_file = random.choice(glob(bg_path + '/*'))

        foreground = resize_image(cv2.imread(fg_file, cv2.IMREAD_UNCHANGED), 512)
        background = resize_image(cv2.imread(bg_file), 512)
        foreground = fit_foreground(background, foreground)

        cv2.imshow('merged', merge(foreground, background))
        cv2.waitKey(0)


def fit_foreground(background, foreground):
    fg_h, fg_w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]
    if bg_h - fg_h > 0:
        foreground = cv2.copyMakeBorder(foreground, bg_h - fg_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        foreground = foreground[0:bg_h, 0: bg_w]
    return foreground


if __name__ == "__main__":
    config = vars(get_arguments())
    composite_images(**config)
