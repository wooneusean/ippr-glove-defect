import numpy as np
import cv2 as cv

from src.detectors.detector_base import Detector
from src.helpers.contour_helper import find_frosting_contour, find_oven_contours


class OvenFrostingDetector(Detector):
    def detect(self):
        oven_contour = find_oven_contours(self.img)
        frosting_contours = find_frosting_contour(self.img)

        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8")

        if oven_contour is None:
            return overlay

        for i, contour in enumerate(frosting_contours):

            x, y, w, h = cv.boundingRect(contour)

            center_x = x + w/2
            center_y = y + h/2
            is_on_glove = cv.pointPolygonTest(
                oven_contour,
                # (center_x, center_y),
                (x, y),
                False
            )

            if is_on_glove:
                cv.rectangle(overlay, (x, y), ((x+w), (y+h)), (0, 255, 255), 3)

            cv.imshow("overlay", overlay)
            cv.waitKey(0)

            message = 'Hole'
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
                (x+w, y+h),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255, 255),
                1,
                cv.LINE_AA,
            )


        cv.imshow("overlay", overlay)
        cv.waitKey(0)
        return overlay


img = cv.imread("../img/oven_frosting.png")
img = cv.resize(img, (500, 500))
OvenFrostingDetector(img).detect()