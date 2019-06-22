import itertools
import math
import cv2
import numpy as np


def hough_transform(img, save_output=True):
    """
    Detect lines in an image

    :param img:
    :param save_output:
    :return: List of lines
    """
    grey = convert_to_greyscale(img, save_output)
    gauss = apply_gaussian_blur(grey, save_output)
    opening = create_opening(gauss, save_output)
    edges = create_canny(opening, save_output)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, 30, 10)
    hough_lines = []
    if lines is not None:
        transformed = [transform_line(line) for line in lines]
        filtered = [line for line in transformed if line is not None]
        for line in filtered:
            endpoints = calculate_endpoints(line, img)
            hough_lines.append(endpoints)

    for line in hough_lines:
        cv2.line(img, line[0], line[1], (0, 0, 255), 2)

    cv2.imwrite('../pictures/output/hough.jpg', img)
    return hough_lines


def calculate_endpoints(line, image):
    """
    Calculate the endpoints which we can use to draw the line on the image
    :param line:
    :param image:
    :return:
    """
    (slope, translation) = line[:2]
    (height, width) = image.shape[:2]
    return (0, int(np.round(translation))), (width, int(np.round(slope * width + translation)))


def transform_line(line):
    """
    Transforms a line into an equation of its slope and its translation
    y = slope * x + translation
    :param line:
    :return:
    """
    (x1, y1, x2, y2) = line[0][:4]
    if x1 == x2:
        return None
    else:
        slope = (y2 - y1) / (x2 - x1)

    translation = y1 - (slope * x1)
    return slope, translation


def convert_to_greyscale(image, save_output=True):
    """
    Convert an image to greyscale
    :param image:
    :param save_output:
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey)
    if save_output:
        cv2.imwrite('../pictures/output/grey.jpg', grey)
    return grey


def apply_gaussian_blur(image, save_output=True):
    """
    Apply a gaussian blur to an image
    :param image:
    :param save_output:
    """
    gauss = cv2.GaussianBlur(image, (5, 5), 0)
    if save_output:
        cv2.imwrite('../pictures/output/gauss.jpg', gauss)
    return gauss


def create_opening(image, save_output=True):
    """
    Create the opening for an image
    :param image:
    :param save_output:
    """
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    if save_output:
        cv2.imwrite('../pictures/output/opening.jpg', opening)
    return opening


def create_canny(image, save_output=True):
    """
    Create the canny for an image
    :param image:
    :param save_output:
    """
    edges = cv2.Canny(image, 40, 150, apertureSize=3)
    if save_output:
        cv2.imwrite('../pictures/output/canny.jpg', edges)
    return edges


def image_resize(image, resized_width=None, resized_height=None, inter=cv2.INTER_AREA):
    """
    Resize an image, keeping the ability to maintain the aspect ratio
    :param image: The image to resize
    :param resized_width: The expected height of the image
    :param resized_height: The expected with of the image
    :param inter: The interpolation mode
    :return: The resized image
    """
    (height, width) = image.shape[:2]

    if resized_width is None and resized_height is None:
        return image

    if resized_width is None:
        ratio = resized_height / height
        dimensions = (int(width * ratio), resized_height)
    elif resized_height is None:
        ratio = resized_width / width
        dimensions = (resized_width, int(height * ratio))
    else:
        dimensions = (resized_width, resized_height)

    return cv2.resize(image, dimensions, interpolation=inter)


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
