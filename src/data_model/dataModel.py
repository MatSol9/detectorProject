from typing import Tuple

import numpy as np
from datetime import datetime


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
