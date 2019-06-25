import argparse
import random
from glob import glob
from os.path import splitext, basename

import cv2

from scene_change import merge
from vanishing_point import resize_image


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('fg_path', help='Path to folder containing foreground')
    parser.add_argument('bg_path', help='Path to folder containing background')

    return parser.parse_args()


def composite_images(fg_path, bg_path):

    for fg_file in glob(fg_path + '/*'):
        bg_file = random.choice(glob(bg_path + '/*'))

        foreground = resize_image(cv2.imread(fg_file, cv2.IMREAD_UNCHANGED), 512)
        foreground = foreground[0:-10, :]
        background = resize_image(cv2.imread(bg_file), 512)
        foreground = fit_foreground(background, foreground)

        merged = merge(foreground, background)
        fg = splitext(basename(fg_file))[0]
        bg = splitext(basename(bg_file))[0]
        cv2.imwrite('output/merged_{}__{}.jpg'.format(fg, bg), merged * 255)


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
