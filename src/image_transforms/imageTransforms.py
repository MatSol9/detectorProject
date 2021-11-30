from typing import List
from typing import Tuple

import cv2
import numpy as np

import src.multi_thread_lib.multiThreadLib as mtl
from src.data_model.dataModel import FrameObject


class GrayscaleTransform(mtl.OperationParent):
    def __init__(self):
        super().__init__()

    def run(self, input_object: List[FrameObject]):
        frame = input_object[0]
        print(frame.get_frame())
        frame_gray = cv2.cvtColor(frame.get_frame(), cv2.COLOR_RGB2GRAY)
        return FrameObject(frame_gray, frame.get_index())


class DetectColoursThresholdsTransform(mtl.OperationParent):
    def __init__(self, thresholds: Tuple[Tuple]):
        super().__init__(thresholds)

    def run(self, input_object: List[FrameObject]):
        frame = input_object[0]
        frame_detected = cv2.inRange(frame.get_frame(), self.get_side_input()[0], self.get_side_input()[1])
        return FrameObject(frame_detected, frame.camera_index)


class ClearDetectedTransform(mtl.OperationParent):
    def __init__(self):
        super().__init__()

    def run(self, input_object: FrameObject):
        frame_cleared = cv2.morphologyEx(input_object.get_frame(), cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        frame_cleared = cv2.morphologyEx(frame_cleared, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return FrameObject(frame_cleared.astype(np.float), input_object.camera_index)
