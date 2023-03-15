import sys

import cv2 as cv
import numpy as np

import latex_hole
from latex_hole import LatexHoleDetector
from latex_stain import LatexStainDetector
from latex_tear import LatexTearDetector

if __name__ == "__main__":
    a = cv.imread("img/blue_glove_hole_1.jpg")

    if a is None:
        sys.exit("No input image")

    # add your detection code here
    hole_result = LatexTearDetector(a).detect()
    tear_result = LatexHoleDetector(a).detect()

    combined_result = np.zeros((a.shape[0], a.shape[1], 4), dtype="uint8")

    # then add the result into this array
    for result in [hole_result, tear_result]:
        combined_result += result

    alpha_foreground = combined_result[:, :, 3] / 255.0
    for color in range(0, 3):
        a[:, :, color] = (1.0 - alpha_foreground) * a[:, :, color] + \
            alpha_foreground * combined_result[:, :, color]

    cv.imshow("Output", a)

    cv.waitKey(0)
