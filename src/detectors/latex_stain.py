import cv2 as cv
import numpy as np

from helpers.contour_helper import find_latex_contour, find_stain_contours


class LatexStainDetector:
    def __init__(self, img):
        self.img = img

    def detect(self):
        # Find latex mask
        latex_contour = find_latex_contour(self.img)

        # Find stain mask
        stain_contours = find_stain_contours(self.img)

        # find min area rect
        minRect = [None]*len(stain_contours)
        minRectangle = [None]*len(stain_contours)
        for i, contour in enumerate(stain_contours):
            minRect[i] = cv.minAreaRect(contour)
            if contour.shape[0] > 5:
                minRectangle[i] = cv.boundingRect(contour)

        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8"
        )

        # draw contours
        for i, contour in enumerate(stain_contours):
            if minRectangle[i] is None:
                continue

            (x, y, w, h) = minRectangle[i]
            aspect_ratio = w / h
            if aspect_ratio < 1 and aspect_ratio > 0:
                aspect_ratio = 1 / aspect_ratio

            latex_contour = cv.convexHull(latex_contour)

            # latex_contour = cv.approxPolyDP(
            #     latex_contour,
            #     0.01*cv.arcLength(latex_contour, True),
            #     True
            # )

            # cv.drawContours(overlay, stain_contours, -1, (0, 255, 0, 255), 2)
            # cv.imshow('stain_overlay', overlay)

            is_within_mask = cv.pointPolygonTest(
                latex_contour,
                (x + w/2, y + h/2),
                False
            )

            area = cv.contourArea(contour, True)
            if aspect_ratio < 2.25 and area > 150 and area < 2000 and is_within_mask >= 0:
            # if area > 100:
                box = cv.boundingRect(contour)

                cv.rectangle(overlay, minRectangle[i], (255, 0, 0, 255), 2)

                # doing this to center the text
                message = f'Stain ({int(area)}), {aspect_ratio:.2f}, {is_within_mask:.2f})'
                # message = 'Stain'
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
                    (255, 0, 0, 255),
                    1,
                    cv.LINE_AA,
                )

        return overlay
