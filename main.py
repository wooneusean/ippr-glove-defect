import sys

import cv2 as cv
import numpy as np

import latex_hole
import latex_tear

if __name__ == "__main__":
    a = cv.imread("img/blue_glove_hole_1.jpg")

    if a is None:
        sys.exit("No input image")

    # add your detection code here
    hole_result = latex_hole.detect(a)
    tear_result = latex_tear.detect(a)

    combined_result = np.zeros((a.shape[0], a.shape[1], 4), dtype="uint8")

    # then add the result into this array
    for result in [hole_result, tear_result]:
        combined_result += result

    alpha_foreground = combined_result[:, :, 3] / 255.0
    for color in range(0, 3):
        a[:, :, color] = (1.0 - alpha_foreground) * a[:, :, color] + \
            alpha_foreground * combined_result[:, :, color]

    cv.imshow("combined_result", a)

    cv.waitKey(0)
