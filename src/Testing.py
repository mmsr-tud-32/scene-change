import sys

import cv2

from vanishing_point import hough_transform, find_intersections, find_vanishing_point, image_resize

filepath = sys.argv[1]

img = image_resize(cv2.imread(filepath), 512)
hough_lines = hough_transform(img)

if not hough_lines:
    print("No lines detected")
    cv2.imshow('Canny', cv2.imread('../pictures/output/canny.jpg'))
    cv2.waitKey(0)
    exit(1)

intersections = find_intersections(hough_lines)

if not intersections:
    print("No intersections detected")
    exit(1)

grid_size = min(img.shape[0], img.shape[1]) // 15
vanishing_point = find_vanishing_point(img, grid_size, intersections)
filename = '../pictures/output/center.jpg'
cv2.imwrite(filename, img)
cv2.imshow('image', img)
cv2.waitKey(0)
