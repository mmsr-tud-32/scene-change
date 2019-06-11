import os, sys

import cv2

from vanishing_point import hough_transform, find_intersections, sample_lines, find_vanishing_point

filepath = sys.argv[1]

print(filepath)
if filepath.endswith(".jpg"):
    print(filepath)
    img = cv2.imread(filepath)
    hough_lines = hough_transform(img)
    print(hough_lines)
    if hough_lines:
        random_sample = sample_lines(hough_lines, 100)
        intersections = find_intersections(random_sample)
        print(intersections)
        if intersections:
            grid_size = min(img.shape[0], img.shape[1]) // 30
            vanishing_point = find_vanishing_point(img, grid_size, intersections)
            filename = '../pictures/output/center' + '.jpg'
            cv2.imwrite(filename, img)
