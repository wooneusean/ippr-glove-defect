from src.detectors.detector_base import Detector
from src.helpers.contour_helper import find_skin_contours, find_cloth_contours


class ClothHoleDetector(Detector):
    def detect(self):
        glove_contours = find_cloth_contours(self.img)

        skin_contours = find_skin_contours(self.img)