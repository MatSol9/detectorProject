from datetime import datetime
from typing import Tuple, List

import cv2
import numpy as np
import yaml


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton
class Config:
    def __init__(self):
        with open("src/resources/config.yml", "r") as ymlfile:
            self.cfg: dict = yaml.load(ymlfile, Loader=yaml.FullLoader)
            morphology_options: dict = {
                "MORPH_ELLIPSE": cv2.MORPH_ELLIPSE,
                "MORPH_RECT": cv2.MORPH_RECT
            }
            self.morphology_options = (morphology_options.get(self.cfg.get("morphology").get("shape")),
                                       tuple(self.cfg.get("morphology").get("size")))

    def get_detection_minimums(self, camera_index: int) -> Tuple[int, int, int]:
        return tuple(self.cfg.get("cameras").get(camera_index).get("detection_minimums"))

    def get_detection_maximums(self, camera_index: int) -> Tuple[int, int, int]:
        return tuple(self.cfg.get("cameras").get(camera_index).get("detection_maximums"))

    def get_morphology_options(self) -> Tuple[int, Tuple[int, int]]:
        return self.morphology_options

    def get_camera_indexes(self) -> List[int]:
        return list(self.cfg.get("cameras").keys())


class FrameObject:
    def __init__(self, frame: np.ndarray, camera_index: int):
        self.frame: np.ndarray = frame
        self.timestamp: float = datetime.now().timestamp()
        self.camera_index: int = camera_index

    def get_frame(self) -> np.ndarray:
        return self.frame

    def get_index(self) -> int:
        return self.camera_index


class FrameObjectWithDetectedCenterOfMass(FrameObject):
    def __init__(self, frame: np.ndarray, camera_index: int, c_x: int, c_y: int, was_detected: bool):
        super(FrameObjectWithDetectedCenterOfMass, self).__init__(frame, camera_index)
        self.c_x = c_x
        self.c_y = c_y
        self.was_detected = was_detected

    def get_center_of_mass(self) -> Tuple[int, int]:
        return self.c_x, self.c_y
