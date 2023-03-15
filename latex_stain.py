import cv2 as cv
import numpy as np


class LatexStainDetector:
    def __init__(self, img):
        self.img = img

    def detect(self):
        overlay = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4),
            dtype="uint8"
        )

        return overlay
