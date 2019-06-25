import argparse
import sys

import cv2

from vanishing_point import resize_image, vanishing_point


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('fg_path', help='Path to image containing foreground')
    parser.add_argument('fg_segmented_path', help='Path to image containing segmented foreground')
    parser.add_argument('bg_path', help='Path to image containing background')

    return parser.parse_args()


def composite_images(fg_path, fg_segmented_path, bg_path):
    foreground = resize_image(cv2.imread(fg_path, cv2.IMREAD_UNCHANGED), 512)
    foreground_segmented = resize_image(cv2.imread(fg_segmented_path, cv2.IMREAD_UNCHANGED), 512)
    background = resize_image(cv2.imread(bg_path), 512)

    vp_foreground = vanishing_point(foreground)
    vp_background = vanishing_point(background)

    cv2.imshow('fg', foreground)
    cv2.imshow('bg', background)

    cv2.waitKey(0)


if __name__ == "__main__":
    config = vars(get_arguments())
    composite_images(**config)
