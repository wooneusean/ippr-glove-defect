import numpy as np
import cv2 as cv

from src.detectors.detector_base import Detector
from src.helpers.contour_helper import find_flour_contour, find_oven_contours


class OvenFlourDetector(Detector):
    def detect(self):
        oven_contour = find_oven_contours(self.img)
        flour_contours = find_flour_contour(self.img)

        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8")

        if flour_contours is None:
            return overlay

        for i, contour in enumerate(flour_contours):

            x, y, w, h = cv.boundingRect(contour)

            center_x = x + w/2
            center_y = y + h/2
            is_on_glove = cv.pointPolygonTest(
                oven_contour,
                (center_x, center_y),
                False
            )

            if is_on_glove:
                cv.rectangle(overlay, (x, y), ((x+w), (y+h)), (0, 255, 0, 255), 2)

            # cv.imshow("overlay", overlay)
            # cv.waitKey(0)

            message = 'Flour'
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
                (x, y+h+text_height+5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0, 255),
                1,
                cv.LINE_AA,
            )


        # cv.imshow("overlay", overlay)
        # cv.waitKey(0)

        return overlay


# img = cv.imread("../img/oven_burn_1.png")
# img = cv.resize(img, (416, 416))
# OvenFlourDetector(img).detect()