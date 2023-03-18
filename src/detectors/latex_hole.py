import cv2 as cv
import numpy as np

from src.detectors.detector_base import Detector
from src.helpers.contour_helper import find_latex_contour, find_skin_contours


class LatexHoleDetector(Detector):
    def detect(self):
        # Find latex mask
        latex_contour = find_latex_contour(self.img)

        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8"
        )

        if latex_contour is None:
            return overlay

        # Find skin mask
        skin_contours = find_skin_contours(self.img)

        # find min area rect
        minRect = [None]*len(skin_contours)
        minEllipse = [None]*len(skin_contours)
        for i, contour in enumerate(skin_contours):
            minRect[i] = cv.minAreaRect(contour)
            if contour.shape[0] > 5:
                minEllipse[i] = cv.fitEllipse(contour)

        # draw contours
        for i, contour in enumerate(skin_contours):
            if minEllipse[i] is None:
                continue

            (x, y), (mx, my), angle = minEllipse[i]
            aspect_ratio = mx / my
            if aspect_ratio < 1 and aspect_ratio > 0:
                aspect_ratio = 1 / aspect_ratio

            latex_contour = cv.convexHull(latex_contour)

            # cv.drawContours(overlay, [latex_contour], -1, (0, 255, 0, 255), 2)
            # cv.imshow('hole_overlay', overlay)

            is_within_mask = cv.pointPolygonTest(
                latex_contour,
                (x, y),
                False
            )

            area = cv.contourArea(contour)
            box = cv.boundingRect(contour)
            bounding_box_area = box[2] * box[3]
            if aspect_ratio < 2 and bounding_box_area > 400 and is_within_mask >= 0:
                # if area > 100:
                cv.rectangle(
                    overlay,
                    box,
                    (0, 0, 255, 255),
                    2
                )

                # doing this to center the text
                # message = f'Hole ({is_within_mask}, {area}, {aspect_ratio}))'
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
