import sys

import cv2 as cv
import numpy as np

from detectors.latex_hole import LatexHoleDetector
from detectors.latex_stain import LatexStainDetector
from detectors.latex_tear import LatexTearDetector

if __name__ == "__main__":
    a = cv.imread("img/blue_glove_tear_3.jpg")

    if a is None:
        sys.exit("No input image")

    # add your detection code here
    hole_result = LatexHoleDetector(a).detect()
    tear_result = LatexTearDetector(a).detect()
    stain_result = LatexStainDetector(a).detect()

    combined_result = np.zeros((a.shape[0], a.shape[1], 4), dtype="uint8")

    # then add the result into this array
    for result in [hole_result, tear_result, stain_result]:
        combined_result += result

    alpha_foreground = combined_result[:, :, 3] / 255.0
    for color in range(0, 3):
        a[:, :, color] = (1.0 - alpha_foreground) * a[:, :, color] + \
            alpha_foreground * combined_result[:, :, color]

    cv.imshow("Output", a)

    cv.waitKey(0)
