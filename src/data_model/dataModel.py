from datetime import datetime
from typing import Tuple, List, Dict

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

    def get_camera_indexes(self) -> List[int]:
        return list(self.cfg.get("cameras").keys())

    def get_max_search_index(self) -> int:
        return int(self.cfg.get("max_search_index"))

    def get_objects(self) -> Dict[int, Dict]:
        return self.cfg.get("objects")

    def get_window_size(self) -> Tuple[int, int]:
        return self.cfg.get("field_of_detection").get("x"), self.cfg.get("field_of_detection").get("y")

    def get_tag_family(self) -> str:
        return self.cfg.get("tag_family")


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
    def __init__(self, frame: np.ndarray, camera_index: int, centers: Dict[int, Tuple[int, int]], rots: Dict):
        super(FrameObjectWithDetectedCenterOfMass, self).__init__(frame, camera_index)
        self.centers = centers
        self.rots = rots
        self.indexes = centers.keys()

    def get_center_of_mass(self, index) -> Tuple[int, int]:
        return self.centers.get(index)

    def get_rotation(self, index):
        return self.rots.get(index)
