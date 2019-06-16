import sys

import cv2

from vanishing_point import hough_transform, find_intersections, sample_lines, find_vanishing_point

filepath = sys.argv[1]

img = cv2.imread(filepath)
hough_lines = hough_transform(img)

if not hough_lines:
    print("No lines detected")
    exit(1)

random_sample = sample_lines(hough_lines, 100)
intersections = find_intersections(random_sample)

if not intersections:
    print("No intersections detected")
    exit(1)

grid_size = min(img.shape[0], img.shape[1]) // 15
vanishing_point = find_vanishing_point(img, grid_size, intersections)
filename = '../pictures/output/center' + '.jpg'
cv2.imwrite(filename, img)
cv2.imshow('image', img)
cv2.waitKey(0)
