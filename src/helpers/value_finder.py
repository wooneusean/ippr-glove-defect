import cv2 as cv
import numpy as np

###############
# This is a helper file to find color values for any image
###############
value_lower     = np.array([65, 115, 110])
value_higher    = np.array([85, 130, 125])

value_thresh = 0


def update_lower(val, col, a):
    value_lower[col] = val
    extracted = cv.inRange(a, value_lower, value_higher)
    cv.imshow("Finder", extracted)


def update_higher(val, col, a):
    value_higher[col] = val
    extracted = cv.inRange(a, value_lower, value_higher)
    cv.imshow("Finder", extracted)


def update(a):
    cv.namedWindow("Finder")
    cv.createTrackbar("X Min", "Finder", value_lower[0], 255,
                      lambda x: update_lower(x, 0, a))
    cv.createTrackbar("X Max", "Finder", value_higher[0], 255,
                      lambda x: update_higher(x, 0, a))
    cv.createTrackbar("Y Min", "Finder", value_lower[1], 255,
                      lambda x: update_lower(x, 1, a))
    cv.createTrackbar("Y Max", "Finder", value_higher[1], 255,
                      lambda x: update_higher(x, 1, a))
    cv.createTrackbar("Z Min", "Finder", value_lower[2], 255,
                      lambda x: update_lower(x, 2, a))
    cv.createTrackbar("Z Max", "Finder", value_higher[2], 255,
                      lambda x: update_higher(x, 2, a))

    extracted = cv.inRange(a, value_lower, value_higher)
    cv.imshow("Finder", extracted)

    cv.waitKey(0)


def update_value_scalar(val, a):
    global value_thresh

    value_thresh = val

    ret, extracted = cv.threshold(
        a,
        value_thresh,
        255,
        cv.THRESH_BINARY
    )
    cv.imshow("Finder", extracted)


def update_scalar(a):
    global value_thresh

    cv.namedWindow("Finder")
    cv.createTrackbar(
        "Min",
        "Finder",
        value_thresh,
        255,
        lambda x: update_value_scalar(x, a)
    )

    ret, extracted = cv.threshold(
        a,
        value_thresh,
        255,
        cv.THRESH_BINARY
    )
    cv.imshow("Finder", extracted)
    cv.waitKey(0)


a = cv.imread("img/blue_glove_stain_2.jpg")
b = cv.imread("img/blue_glove_hole_3.jpg")
c = np.hstack((a, b))
cv.imshow("Original c", c)

c_lab = cv.cvtColor(c, cv.COLOR_BGR2LAB)
update(c_lab)
