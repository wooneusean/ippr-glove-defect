import cv2 as cv
import numpy as np

from detectors.detector_base import Detector
from helpers.contour_helper import find_frosting_contour, find_oven_contours


class OvenFrostingDetector(Detector):
    def detect(self):
        oven_contour = find_oven_contours(self.img)
        frosting_contours = find_frosting_contour(self.img)

        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8")

        if frosting_contours is None:
            return overlay

        message = 'Frosting'

        for i, glove in enumerate(oven_contour):
            for j, contour in enumerate(frosting_contours):
                moments = cv.moments(contour)
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                x, y, w, h = cv.boundingRect(contour)

                is_on_glove = cv.pointPolygonTest(
                    glove,
                    (cX, cY),
                    False
                )

                if is_on_glove > 0:
                    cv.rectangle(overlay, (x, y), ((x + w), (y + h)), (180, 105, 255, 255), 2)

                    cv.putText(
                        overlay,
                        message,
                        (x, y - 5),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (180, 105, 255, 255),
                        1,
                        cv.LINE_AA,
                    )

        # cv.imshow("overlay", overlay)
        # cv.waitKey(0)

        return overlay


# img = cv.imread("../img/burn_3.png")
# img = cv.resize(img, (500, 500))
# OvenFrostingDetector(img).detect()