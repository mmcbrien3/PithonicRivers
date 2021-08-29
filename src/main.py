import os
import sys

import cv2 as cv
import numpy as np


def flatten_image_to_bw(river_img):
    """
    This function takes in a 3-channel image of a river and returns
    a 1-channel, black & white image of the river.
    :param river_img: the input image
    :return: the flattened, black & white image
    """
    grayed_img = cv.cvtColor(river_img, cv.COLOR_BGR2GRAY)
    ret, thresholded_img = cv.threshold(grayed_img, 200, 255, cv.THRESH_BINARY_INV)
    return thresholded_img


def thin_river(river_img):
    """
    This function takes in a black & white image of a river and "thins" it.
    Thinning is the process of reducing of curve to a minimum width. The
    purpose of this function is to find the center-line of the river, which
    makes later calculations simpler.
    :param river_img: the input image, black & white
    :return: thinned image
    """
    thinned = cv.ximgproc.thinning(river_img)
    return thinned


def find_contour(river_img):
    """
    This function uses built in opencv tools to find the contours of the input
    image. The purpose is to find contour of a pre-thinned river, which can
    then be used later for calculating the sinuosity.
    :param river_img: the input image
    :return: the primary contour of the river
    """
    contour = cv.findContours(
        river_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0][0]
    return contour


def find_sinuosity(contour, input_shape):
    """
    This function takes in a contour and the shape of the original input image
    to determine the sinuosity. The input shape is required to determine
    the start and end point of the contour (this assumes that the
    start and end point are on the edges of the input image).
    :param contour: the contour to find the sinuosity of
    :param input_shape: the shape of the image from which the contour was found
    :return: the sinuosity of the contour
    """
    arc_length = cv.arcLength(contour, False) / 2
    first_coordinate, last_coordinate = get_start_and_end_of_river(contour, input_shape)
    direct_length = distance_between_points(first_coordinate, last_coordinate)

    return arc_length / direct_length


def get_start_and_end_of_river(contour, input_shape):
    """
    This function takes in a contour and returns the two pixel locations
    that represent the beginning and end of the contour. This assumes
    that the contour's beginning and end are at the edge of the image
    and only the start and end points of the contour are on the edge
    :param contour: the contour to find the start and end of
    :param input_shape: the shape of the image from which the contour was found
    :return: a tuple holding two values representing the start coordinate and
    the end coordinate of the input contour
    """
    points_on_edge = contour[
        (contour[:, 0, 0] == 0) |
        (contour[:, 0, 1] == 0) |
        (contour[:, 0, 0] == input_shape[1] - 1) |
        (contour[:, 0, 1] == input_shape[0] - 1)
    ]
    distance_grid = np.zeros((points_on_edge.size, points_on_edge.size))
    for st_idx, st in enumerate(points_on_edge):
        for end_idx, end in enumerate(points_on_edge):
            distance_grid[st_idx][end_idx] = distance_between_points(st[0], end[0])
    max_index = np.where(distance_grid == np.max(distance_grid))

    first_coordinate = points_on_edge[max_index[0][0]][0]
    last_coordinate = points_on_edge[max_index[1][0]][0]
    return first_coordinate, last_coordinate


def distance_between_points(point_a, point_b):
    """
    This function takes in two pixel locations and uses the pythagorean
    theorem to calculate the distance between the points
    :param point_a: first input point
    :param point_b: second input point
    :return: pythagorean distance
    """
    return (
                (point_a[0] - point_b[0])**2 +
                (point_a[1] - point_b[1])**2
            ) ** 0.5


def visualize_result(input_img, bw_img, thin_img, sinuosity_val):
    """
    This function can be used to output an image that shows how the final
    sinuosity value was determined
    :param input_img: the original input image of the river to find the
    sinuosity of
    :param bw_img: the flattened, black & white image of the river
    :param thin_img: the thinned image of the river
    :param sinuosity_val: the final, calculated sinuosity value
    :return: None
    """
    sinuosity_image = np.zeros(input_img.shape, dtype=np.uint8)
    fc, lc = get_start_and_end_of_river(river_contour, bw_img.shape)
    sinuosity_image = cv.drawContours(sinuosity_image, river_contour, -1,
                                      (255, 0, 0), thickness=3)
    sinuosity_image = cv.line(sinuosity_image, fc, lc, (0, 0, 255), thickness=3)

    view_bw_image = np.stack((bw_img,) * 3, axis=-1)
    view_thin_image = np.stack((thin_img,) * 3, axis=-1)
    result_image = cv.hconcat(
        [input_img, view_bw_image, view_thin_image, sinuosity_image])
    cv.imshow(f"Calculated Sinuosity: {sinuosity_val}", result_image)
    cv.waitKey(0)


if __name__ == "__main__":
    try:
        input_img_path = sys.argv[1]
    except Exception as e:
        print("Failed to parse input image. Please provide input image path "
              "as first argument of script.", e)
        sys.exit(1)

    should_visualize_result = True
    print(f"Using input image located at: "
          f"{os.path.join(os.getcwd(), input_img_path)}")

    input_img = cv.imread(input_img_path)
    bw_img = flatten_image_to_bw(input_img)
    thin_img = thin_river(bw_img)
    river_contour = find_contour(thin_img)
    sinuosity = find_sinuosity(river_contour, bw_img.shape)

    if should_visualize_result:
        visualize_result(input_img, bw_img, thin_img, sinuosity)

    print(f"Found sinuosity of {sinuosity}")
