import cv2 as cv
import numpy as np

from detectors.detector_base import Detector

class LeatherScratchDetector(Detector):
    def detect(self):
        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8"
        )

        img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        img = cv.convertScaleAbs(img, alpha=1.5, beta=6)

        value_lower = np.array([31, 8, 170])
        value_higher = np.array([74, 30, 255])

        thresh = cv.inRange(img, value_lower, value_higher)

        lines = cv.HoughLinesP(thresh, 1, np.pi / 180, threshold=17, minLineLength=15, maxLineGap=3)

        if lines is None:
            return overlay

        # Create empty mask for contours
        contour_mask = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8"
        )

        # Draw lines on contour mask
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(contour_mask, (x1, y1), (x2, y2), 255, 2)

        contour_mask = cv.cvtColor(contour_mask, cv.COLOR_BGR2GRAY)

        # Find contours on contour mask
        contours, hierarchy = cv.findContours(contour_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # find min area rect
        minRectangle = [None] * len(contours)
        for i, contour in enumerate(contours):
            minRectangle[i] = cv.boundingRect(contour)

        for i, contour in enumerate(contours):
            if minRectangle[i] is None:
                continue

            # Find the bounding box
            bbox = cv.boundingRect(contour)

            # Draw the bounding box on the image
            cv.rectangle(overlay, minRectangle[i], (0, 255, 0, 255), 2)

            message = 'Scratch'
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
                (0, 255, 0, 255),
                1,
                cv.LINE_AA,
            )

        return overlay
