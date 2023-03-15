import cv2 as cv
import numpy as np

###############
# This is a helper file to find color values for any image
###############
value_lower = np.array([0, 0, 0])
value_higher = np.array([255, 255, 255])


def updateLower(val, col, a):
    value_lower[col] = val
    extracted = cv.inRange(a, value_lower, value_higher)
    cv.imshow("Finder", extracted)
    pass


def updateHigher(val, col, a):
    value_higher[col] = val
    extracted = cv.inRange(a, value_lower, value_higher)
    cv.imshow("Finder", extracted)
    pass


def update(a):
    cv.namedWindow("Finder")
    cv.createTrackbar("X Min", "Finder", value_lower[0], 255,
                      lambda x: updateLower(x, 0, a))
    cv.createTrackbar("X Max", "Finder", value_higher[0], 255,
                      lambda x: updateHigher(x, 0, a))
    cv.createTrackbar("Y Min", "Finder", value_lower[1], 255,
                      lambda x: updateLower(x, 1, a))
    cv.createTrackbar("Y Max", "Finder", value_higher[1], 255,
                      lambda x: updateHigher(x, 1, a))
    cv.createTrackbar("Z Min", "Finder", value_lower[2], 255,
                      lambda x: updateLower(x, 2, a))
    cv.createTrackbar("Z Max", "Finder", value_higher[2], 255,
                      lambda x: updateHigher(x, 2, a))

    extracted = cv.inRange(a, value_lower, value_higher)
    cv.imshow("Finder", extracted)

    cv.waitKey(0)


a = cv.imread("img/blue_glove_tear_1.jpg")
a_hsv = cv.cvtColor(a, cv.COLOR_BGR2HSV)
update(a_hsv)
