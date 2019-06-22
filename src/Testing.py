import sys

import cv2

from vanishing_point import hough_transform, find_all_intersections, find_vanishing_point, resize_image

filepath = sys.argv[1]

img = resize_image(cv2.imread(filepath), 512)
hough_lines = hough_transform(img)

if not hough_lines:
    print("No lines detected")
    cv2.imshow('Canny', cv2.imread('../pictures/output/canny.jpg'))
    cv2.waitKey(0)
    exit(1)

intersections = find_all_intersections(hough_lines)

for x,y in intersections:
    print(x, y)
    cv2.line(img, (x, y), (x, y), (0, 255, 0), 10)

if not intersections:
    print("No intersections detected")
    exit(1)

grid_size = min(img.shape[0], img.shape[1]) // 10
vanishing_point = find_vanishing_point(img, grid_size, intersections)
filename = '../pictures/output/center.jpg'
cv2.imwrite(filename, img)
cv2.imshow('image', img)
cv2.waitKey(0)
