import cv2 as cv
import numpy as np


def detect(a):
    a_lab = cv.cvtColor(a, cv.COLOR_BGR2LAB)
    a_lab_gray = cv.cvtColor(a_lab, cv.COLOR_BGR2GRAY)
    a_gauss = cv.GaussianBlur(a_lab_gray, (5, 5), 0)
    (thresh, a_thresh) = cv.threshold(
        a_gauss,
        0,
        255,
        cv.THRESH_OTSU
    )
    a_dilate = cv.dilate(a_thresh, None, iterations=2)
    a_erode = cv.erode(a_dilate, None, iterations=2)
    a_mask = cv.bitwise_not(a_erode)
    # a_mask_points = cv.findNonZero(a_mask)
    # a_mask_box = cv.boundingRect(a_mask_points)
    # a_out = cv.rectangle(a, a_mask_box, (0, 255, 0), 2)

    # find contours
    a_canny = cv.Canny(a_mask, 100, 200)
    cv.imshow("canny", a_canny)

    a_contours, _ = cv.findContours(
        a_canny,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in a_contours:
        area = cv.contourArea(contour)
        box = cv.boundingRect(contour)
        if area > 250:
            cv.rectangle(a, box, (0, 255, 0), 2)
            cv.putText(
                a,
                f'Hole (Area={str(area)})',
                (box[0], box[1] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

    # cv.drawContours(a, contour_list, -1, (0, 255, 0), 2)
    return a