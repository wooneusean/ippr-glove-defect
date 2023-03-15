import cv2 as cv
import numpy as np


def detect(a):
    a_lab = cv.cvtColor(a, cv.COLOR_BGR2LAB)

    # Find latex mask
    latex_lower = np.array([0, 0, 0])
    latex_higher = np.array([255, 120, 255])
    a_hsv = cv.cvtColor(a, cv.COLOR_BGR2HSV)
    latex_extracted = cv.inRange(a_hsv, latex_lower, latex_higher)
    latex_extracted = cv.bitwise_not(latex_extracted, latex_extracted)
    latex_extracted = cv.erode(latex_extracted, None, iterations=2)
    latex_extracted = cv.dilate(latex_extracted, None, iterations=4)

    latex_contours, _ = cv.findContours(
        latex_extracted,
        cv.RETR_CCOMP,
        cv.CHAIN_APPROX_SIMPLE
    )
    latex_extracted = cv.bitwise_not(latex_extracted)
    largest_contour = None
    largest_contour_area = -1
    for cnt in latex_contours:
        current_contour_area = cv.contourArea(cnt)

        if largest_contour is None:
            largest_contour = cnt
        elif current_contour_area > largest_contour_area:
            largest_contour = cnt
            largest_contour_area = cv.contourArea(cnt)

        cv.drawContours(latex_extracted, [cnt], -1, (0, 255, 0), 3)

    latex_extracted = cv.bitwise_not(latex_extracted)
    # cv.imshow("latex_extracted", latex_extracted)

    # Find skin mask
    skin_lower = np.array([120, 130, 130])
    skin_higher = np.array([255, 255, 255])
    skin_extracted = cv.inRange(a_lab, skin_lower, skin_higher)
    # cv.imshow("skin_extracted", skin_extracted)

    a_erode = cv.erode(skin_extracted, None, iterations=2)
    # cv.imshow("a_erode", a_erode)

    a_dilate = cv.dilate(a_erode, None, iterations=2)
    # cv.imshow("a_dilate", a_dilate)

    a_fill = cv.morphologyEx(a_dilate, cv.MORPH_CLOSE, None)
    # cv.imshow("a_fill", a_fill)

    a_canny = cv.Canny(a_fill, 0, 255)
    # cv.imshow("a_canny", a_canny)

    # find contours
    a_contours, _ = cv.findContours(
        a_canny,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    # find min area rect
    minRect = [None]*len(a_contours)
    minEllipse = [None]*len(a_contours)
    for i, c in enumerate(a_contours):
        minRect[i] = cv.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)

    overlay = np.zeros((a.shape[0], a.shape[1], 4), dtype="uint8")

    # draw contours
    for contour, i in zip(a_contours, range(len(a_contours))):
        area = cv.contourArea(contour)
        box = cv.boundingRect(contour)
        (x, y), (mx, my), angle = minEllipse[i]
        aspect_ratio = mx / my
        if aspect_ratio < 1 and aspect_ratio > 0:
            aspect_ratio = 1 / aspect_ratio

        is_within_mask = cv.pointPolygonTest(largest_contour, (x, y), False)

        if aspect_ratio > 2 and area > 200 and is_within_mask >= 0:
            cv.ellipse(overlay, minEllipse[i], (0, 0, 255, 255), 2)

            # doing this to center the text
            message = f'Tear (a: {area:.2f}, r: {aspect_ratio:.2f})'
            text_size, _ = cv.getTextSize(
                message,
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            text_width, _ = text_size
            cv.putText(
                overlay,
                message,
                (int(box[0] - (text_width / 2)), box[1] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255, 255),
                1,
                cv.LINE_AA,
            )

    return overlay
