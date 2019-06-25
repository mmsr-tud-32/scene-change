import sys

import cv2

from vanishing_point import resize_image, vanishing_point

filepath = sys.argv[1]

img = resize_image(cv2.imread(filepath), 512)

vanishing_point = vanishing_point(img, save_output=True)

cv2.imshow('vp', img)
cv2.waitKey(0)
