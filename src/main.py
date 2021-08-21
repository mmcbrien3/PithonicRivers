import sys
import numpy as np
import cv2 as cv


def outline_river(input_img):
    """
    Takes in an image of a river and outlines the edges (contours)
    """
    cv.imshow("input image", input_img)

    imgray = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
    ret, im = cv.threshold(imgray, 200, 255, cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contoured_im = cv.drawContours(input_img,  contours, -1, (0,255,0), 2)
    cv.imshow("imgray", imgray)
    cv.imshow("thresholded", im)
    cv.imshow("contours", contoured_im)

    cv.waitKey(0)

def find_sinuosity(river):
    """
    Takes in a contour map and returns the sinuosity value
    """
    pass

if __name__ == "__main__":
    try:
        input_img_path = sys.argv[1]
    except Exception as e:
        print("Failed to parse input image. Please provide input image path "\
                "as first argument of script.", e)
        sys.exit(1)
    
    print(f"Using input image located at: {input_img_path}")

    input_img = cv.imread(input_img_path)
    river = outline_river(input_img)
    sinuosity = find_sinuosity(river)