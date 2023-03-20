import cv2 as cv
import numpy as np

from detectors.detector_base import Detector

class LeatherMouldDetector(Detector):
    def detect(self):
        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8"
        )

        img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        img = cv.convertScaleAbs(img, alpha=1, beta=6)

        # Thresholding
        value_lower = np.array([0, 0, 132])
        value_higher = np.array([160, 11, 240])

        thresh = cv.inRange(img, value_lower, value_higher)
        thresh = cv.bitwise_not(thresh, thresh)

        # Morphological operations
        kernel = np.ones((7, 7), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        # Get Contours
        contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if contours is None:
            return overlay

        # find min area rect
        minRectangle = [None]*len(contours)
        for i, contour in enumerate(contours):
                minRectangle[i] = cv.boundingRect(contour)

        for i, contour in enumerate(contours):
            if minRectangle[i] is None:
                continue

            area = cv.contourArea(contour, True)
            if area > 150:
                # Find the bounding box
                bbox = cv.boundingRect(contour)

                # Draw the bounding box on the image
                cv.rectangle(overlay, minRectangle[i], (255, 0, 0, 255), 2)

                message = 'Mould'
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
                    (int(bbox[0] - (text_width / 2)),
                        (bbox[1] + bbox[3]) + text_height + 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0, 255),
                    1,
                    cv.LINE_AA,
                )

        return overlay
