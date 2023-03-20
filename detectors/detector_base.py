from abc import abstractmethod


class Detector():
    img = None

    def __init__(self, img) -> None:
        self.img = img

    @abstractmethod
    def detect(self):
        pass
