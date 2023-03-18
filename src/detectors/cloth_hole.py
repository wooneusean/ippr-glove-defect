import numpy as np
import cv2 as cv

from src.detectors.detector_base import Detector
from src.helpers.contour_helper import find_skin_contours, find_oven_contours


class ClothHoleDetector(Detector):
    def detect(self):
        cloth_contour = find_oven_contours(self.img)

        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8")

        if cloth_contour is None:
            return overlay

        skin_contours = find_skin_contours(self.img)

        minRect = [None] * len(skin_contours)
        minEllipse = [None] * len(skin_contours)
        for i, contour in enumerate(skin_contours):
            minRect[i] = cv.minAreaRect(contour)
            if contour.shape[0] > 5:
                minEllipse[i] = cv.fitEllipse(contour)

        for i, contour in enumerate(skin_contours):
            if minEllipse[i] is None:
                continue

            (x, y), (mx, my), angle = minEllipse[i]
            aspect_ratio = mx / my
            if aspect_ratio < 1 and aspect_ratio > 0:
                aspect_ratio = 1 / aspect_ratio

            cloth_contour = cv.convexHull(cloth_contour)

            is_within_mask = cv.pointPolygonTest(
                cloth_contour,
                (x, y),
                False
            )

            area = cv.contourArea(contour)
            box = cv.boundingRect(contour)
            bounding_box_area = box[2] * box[3]
            if aspect_ratio < 2 and bounding_box_area > 400 and is_within_mask >= 0:
                cv.rectangle(
                    overlay,
                    box,
                    (0, 0, 255, 255),
                    2
                )

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
                (int(box[0] - (text_width / 2)),
                 (box[1] + box[3]) + text_height + 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255, 255),
                1,
                cv.LINE_AA,
            )

        return overlay

