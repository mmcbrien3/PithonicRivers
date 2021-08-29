import sys
import os
import cv2 as cv
import numpy as np
import itertools


def flatten_image_to_bw(river_img):
    grayed_img = cv.cvtColor(river_img, cv.COLOR_BGR2GRAY)
    ret, thresholded_img = cv.threshold(grayed_img, 200, 255, cv.THRESH_BINARY_INV)
    return thresholded_img


def thin_river(river_img):
    thinned = cv.ximgproc.thinning(river_img)
    cv.imshow("thinned", thinned)
    return thinned


def find_contour(river_img):
    """
    Takes in an image of a river and outlines the edges (contours)
    """
    contours = cv.findContours(
        river_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    contours = np.asarray(list(itertools.chain.from_iterable(contours)))
    return contours


def find_sinuosity(contour, input_shape):
    """
    Takes in a contour map and returns the sinuosity value
    """
    arc_length = cv.arcLength(contour, False) / 2
    first_coordinate, last_coordinate = get_start_and_end_of_river(contour, input_shape)
    direct_length = distance_between_points(first_coordinate, last_coordinate)

    return arc_length / direct_length


def get_start_and_end_of_river(contour, input_shape):
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


def distance_between_points(point_A, point_B):
    # Pythagorean theorem
    return (
                (point_A[0] - point_B[0])**2 +
                (point_A[1] - point_B[1])**2
            ) ** 0.5


if __name__ == "__main__":
    try:
        input_img_path = sys.argv[1]
    except Exception as e:
        print("Failed to parse input image. Please provide input image path "
              "as first argument of script.", e)
        sys.exit(1)

    visualize_result = True
    print(f"Using input image located at: "
          f"{os.path.join(os.getcwd(), input_img_path)}")

    input_img = cv.imread(input_img_path)
    bw_img = flatten_image_to_bw(input_img)
    thin_img = thin_river(bw_img)
    river_contour = find_contour(thin_img)
    sinuosity = find_sinuosity(river_contour, bw_img.shape)

    if visualize_result:
        view_image = np.zeros(input_img.shape)
        fc, lc = get_start_and_end_of_river(river_contour, bw_img.shape)
        view_image = cv.drawContours(view_image, river_contour, -1, (255, 0, 0), thickness=3)
        view_image = cv.line(view_image, fc, lc, (0, 0, 255), thickness=3)
        cv.imshow("input", input_img)
        cv.imshow("result", view_image)
        cv.waitKey(0)

    print(f"Found sinuosity of {sinuosity}")
