import cv2 as cv
import numpy as np

from detectors.detector_base import Detector
from helpers.contour_helper import find_flour_contour, find_oven_contours


class OvenFlourDetector(Detector):
    def detect(self):
        oven_contour = find_oven_contours(self.img)
        flour_contours = find_flour_contour(self.img)

        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8")

        if flour_contours is None:
            return overlay

        message = 'Flour'

        for i, glove in enumerate(oven_contour):
            for j, contour in enumerate(flour_contours):
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
                    cv.rectangle(overlay, (x, y), ((x + w), (y + h)), (255, 0, 0, 255), 2)

                    cv.putText(
                        overlay,
                        message,
                        (x, y - 5),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0, 255),
                        1,
                        cv.LINE_AA,
                    )

        # cv.imshow("overlay", overlay)
        # cv.waitKey(0)

        return overlay


# img = cv.imread("../img/burn_2.png")
# img = cv.resize(img, (500, 500))
# OvenFlourDetector(img).detect()