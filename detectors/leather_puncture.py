import cv2 as cv
import numpy as np

from detectors.detector_base import Detector

class LeatherPunctureDetector(Detector):
    def detect(self):
        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8"
        )

        img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        img = cv.convertScaleAbs(img, alpha=1.5, beta=6)

        # Thresholding
        value_lower = np.array([25, 166, 87])
        value_higher = np.array([255, 255, 253])

        thresh = cv.inRange(img, value_lower, value_higher)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        # Increase size of hole
        dilation = cv.dilate(closing, kernel, iterations=5)

        # Find contours in the image
        contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if contours is None:
            return overlay

        # find min area rect
        minRectangle = [None]*len(contours)
        for i, contour in enumerate(contours):
            minRectangle[i] = cv.boundingRect(contour)

        for i, contour in enumerate(contours):
            if minRectangle[i] is None:
                continue

            area = cv.contourArea(contour, False)
            if 10 < area < 550:
                # Find the bounding box
                bbox = cv.boundingRect(contour)

                # Draw the bounding box on the image
                cv.rectangle(overlay, minRectangle[i], (0, 0, 255, 255), 2)

                message = 'Puncture'
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
                    (0, 0, 255, 255),
                    1,
                    cv.LINE_AA,
                )

        return overlay
