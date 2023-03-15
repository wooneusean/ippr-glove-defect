
import sys

import cv2 as cv
import numpy as np

import hole

if __name__ == "__main__":
    a = cv.imread("img/blue_glove_hole_2.jpg")

    if a is None:
        sys.exit("No input image")

    a = hole.detect(a)

    cv.imshow("out", a)
    cv.waitKey(0)
