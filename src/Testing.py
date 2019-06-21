import sys

import cv2

from vanishing_point import hough_transform, find_intersections, find_vanishing_point, image_resize

filepath = sys.argv[1]

img = cv2.imread(filepath)
resized = image_resize(img, 512)
hough_lines = hough_transform(resized)

if not hough_lines:
    print("No lines detected")
    cv2.imshow('Canny', cv2.imread('../pictures/output/canny.jpg'))
    cv2.waitKey(0)
    exit(1)

intersections = find_intersections(hough_lines)

if not intersections:
    print("No intersections detected")
    exit(1)

grid_size = min(resized.shape[0], resized.shape[1]) // 15
vanishing_point = find_vanishing_point(resized, grid_size, intersections)
filename = '../pictures/output/center.jpg'
cv2.imwrite(filename, resized)
cv2.imshow('image', resized)
cv2.waitKey(0)
