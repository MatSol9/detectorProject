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
