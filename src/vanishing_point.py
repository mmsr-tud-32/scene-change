import itertools
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
        transformed = {transform_line(line) for line in lines}
        filtered = {line for line in transformed if line is not None}
        hough_lines = list(filtered)

    print(hough_lines)
    for line in hough_lines:
        endpoints = calculate_endpoints(line, img)
        cv2.line(img, endpoints[0], endpoints[1], (0, 0, 255), 2)

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


def resize_image(image, resized_width=None, resized_height=None, inter=cv2.INTER_AREA):
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


def find_intersection_point(line1, line2):
    """
    Find the intersection between two lines

    :param line1:
    :param line2:
    :return: The x and y coordinates of the intersection, or None if the lines don't intersect
    """
    (slope_1, translation_1) = line1[:2]
    (slope_2, translation_2) = line2[:2]
    if slope_1 == slope_2:
        return None

    x_coord = (translation_1 - translation_2) / (slope_1 - slope_2)
    y_coord = slope_1 * x_coord + translation_1

    return int(np.round(x_coord)), int(np.round(y_coord))


def find_all_intersections(lines):
    """
    Find all intersections over all lines
    :param lines:
    :return:
    """
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if not line_1 == line_2:
                intersection = find_intersection_point(line_1, line_2)
                if intersection:
                    intersections.append(intersection)

    return intersections


def find_vanishing_point(img, grid_size, intersections):
    """
    Find the vanishing point
    
    :param img:
    :param grid_size:
    :param intersections:
    :return:
    """
    (image_height, image_width) = img.shape[:2]

    grid_rows = (image_height // grid_size) + 1
    grid_columns = (image_width // grid_size) + 1

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
