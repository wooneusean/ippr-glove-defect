import cv2 as cv
import numpy as np

from box_helper import is_within_bb
from detector_base import Detector


class LatexTearDetector(Detector):
    def detect(self):
        a_lab = cv.cvtColor(self.img, cv.COLOR_BGR2LAB)

        # Find latex mask
        largest_bounding_box = self.find_latex_bounding_box(self.img)

        # Find skin mask
        a_contours = self.find_skin_contours(a_lab)

        # find min area rect
        minRect = [None]*len(a_contours)
        minEllipse = [None]*len(a_contours)
        for i, contour in enumerate(a_contours):
            minRect[i] = cv.minAreaRect(contour)
            if contour.shape[0] > 5:
                minEllipse[i] = cv.fitEllipse(contour)

        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8"
        )

        # draw contours
        for i, contour in enumerate(a_contours):
            area = cv.contourArea(contour)
            box = cv.boundingRect(contour)
            (x, y), (mx, my), angle = minEllipse[i]
            aspect_ratio = mx / my
            if aspect_ratio < 1 and aspect_ratio > 0:
                aspect_ratio = 1 / aspect_ratio

            is_within_mask = is_within_bb(
                largest_bounding_box,
                x,
                y
            )

            if aspect_ratio > 2.5 and area > 200 and is_within_mask >= 0:
                cv.ellipse(overlay, minEllipse[i], (0, 0, 255, 255), 2)

                # doing this to center the text
                message = f'Tear'
                text_size, _ = cv.getTextSize(
                    message,
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )
                text_width, text_height = text_size
                cv.putText(
                    overlay,
                    message,
                    (int(box[0] + (text_width / 2)),
                     (box[1] + box[3]) + text_height + 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255, 255),
                    1,
                    cv.LINE_AA,
                )

        return overlay

    def find_skin_contours(self, img):
        skin_lower = np.array([120, 130, 130])
        skin_higher = np.array([255, 255, 255])
        skin_extracted = cv.inRange(img, skin_lower, skin_higher)
        # cv.imshow("skin_extracted", skin_extracted)

        a_erode = cv.erode(skin_extracted, None, iterations=2)
        # cv.imshow("a_erode", a_erode)

        a_dilate = cv.dilate(a_erode, None, iterations=2)
        # cv.imshow("a_dilate", a_dilate)

        a_fill = cv.morphologyEx(a_dilate, cv.MORPH_CLOSE, None)
        # cv.imshow("a_fill", a_fill)

        a_canny = cv.Canny(a_fill, 0, 255)
        # cv.imshow("a_tear_canny", a_canny)

        # find contours
        a_contours, _ = cv.findContours(
            a_canny,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        return a_contours

    def find_latex_bounding_box(self, img):
        latex_lower = np.array([0, 0, 0])
        latex_higher = np.array([255, 120, 255])
        a_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        latex_extracted = cv.inRange(a_hsv, latex_lower, latex_higher)
        latex_extracted = cv.bitwise_not(latex_extracted, latex_extracted)
        latex_extracted = cv.erode(latex_extracted, None, iterations=1)
        latex_extracted = cv.dilate(latex_extracted, None, iterations=1)

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
        # cv.imshow("latex_tear_extracted", latex_extracted)
        return cv.boundingRect(largest_contour)
