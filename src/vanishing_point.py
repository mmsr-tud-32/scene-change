import itertools
import random
import math
from itertools import starmap

import cv2
import numpy as np


# Perform edge detection
def hough_transform(img):
    """
    Detect lines in an image

    :param img:
    :return: List of lines
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    gray = cv2.equalizeHist(gray)
    cv2.imwrite('../pictures/output/gray.jpg', gray)
    kernel = np.ones((5, 5), np.uint8)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite('../pictures/output/gauss.jpg', gauss)

    opening = cv2.morphologyEx(gauss, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
    cv2.imwrite('../pictures/output/opening.jpg', opening)
    edges = cv2.Canny(opening, 40, 150, apertureSize=3)  # Canny edge detection
    cv2.imwrite('../pictures/output/canny.jpg', edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, 50, 50)  # Hough line detection

    hough_lines = []
    # Lines are represented by rho, theta; converted to endpoint notation
    if lines is not None:
        for line in lines:
            new_line = list([(line[0][0], line[0][1]), (line[0][2], line[0][3])])
            angle = abs(math.degrees(math.atan2(new_line[0][0] - new_line[1][0], new_line[0][1] - new_line[1][1])))
            if (angle >= 94 or angle <= 86):# and angle >= 4 and (angle <= 176 or angle >= 184):
                hough_lines.append(extend(new_line))

    for line in hough_lines:
        cv2.line(img, line[0], line[1], (0, 0, 255), 2)

    cv2.imwrite('pictures/output/hough.jpg', img)
    return hough_lines


def extend(line):
    """
    Extend a line to cover the entire image.

    :param line: List[Tuple[Int, Int]]
    :return:
    """
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]
    theta = abs(math.atan2(x1 - x2, y1 - y2))
    a = np.cos(theta)
    b = np.sin(theta)
    new_x1 = int(x1 + 1000 * (-b))
    new_y1 = int(y1 + 1000 * a)
    new_x2 = int(x2 - 1000 * (-b))
    new_y2 = int(y2 - 1000 * a)
    return list([(new_x1, new_y1), (new_x2, new_y2)])


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


# Find intersection point of two lines (not segments!)
def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None  # Lines don't cross

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return x, y


# Find intersections between multiple lines (not line segments!)
def find_intersections(lines):
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if not line_1 == line_2:
                intersection = line_intersection(line_1, line_2)
                if intersection:  # If lines cross, then add
                    intersections.append(intersection)

    return intersections


# Given intersections, find the grid where most intersections occur and treat as vanishing point
def find_vanishing_point(img, grid_size, intersections):
    # Image dimensions
    image_height = img.shape[0]
    image_width = img.shape[1]

    # Grid dimensions
    grid_rows = (image_height // grid_size) + 1
    grid_columns = (image_width // grid_size) + 1

    # Current cell with most intersection points
    max_intersections = 0
    best_cell = (0.0, 0.0)

    for i, j in itertools.product(range(grid_rows), range(grid_columns)):
        cell_left = j * grid_size
        cell_right = (j + 1) * grid_size
        cell_bottom = i * grid_size
        cell_top = (i + 1) * grid_size
        # cv2.rectangle(img, (cell_left, cell_bottom), (cell_right, cell_top), (0, 0, 255), 2)

        current_intersections = 0  # Number of intersections in the current cell
        for x, y in intersections:
            if cell_left < x < cell_right and cell_bottom < y < cell_top:
                current_intersections += 1

        # Current cell has more intersections that previous cell (better)
        if current_intersections > max_intersections:
            max_intersections = current_intersections
            best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)
            print("Best Cell:", best_cell)

    if best_cell[0] != None and best_cell[1] != None:
        rx1 = int(best_cell[0] - grid_size / 2)
        ry1 = int(best_cell[1] - grid_size / 2)
        rx2 = int(best_cell[0] + grid_size / 2)
        ry2 = int(best_cell[1] + grid_size / 2)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.imwrite('pictures/output/center.jpg', img)

    return best_cell
