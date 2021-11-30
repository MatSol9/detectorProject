from typing import List
from typing import Tuple

import cv2

import src.multi_thread_lib.multiThreadLib as mtl
from src.data_model.dataModel import FrameObject
from src.data_model.dataModel import FrameObjectWithDetectedCenterOfMass


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
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
        frame_cleared = cv2.morphologyEx(frame_cleared, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
        return FrameObject(frame_cleared, input_object.camera_index)


class GetMomentsTransform(mtl.OperationParent):
    def __init__(self):
        super().__init__()
        self.c_x = 0
        self.c_y = 0

    def run(self, input_object):
        m = cv2.moments(input_object.get_frame())
        detected = False
        if m["m00"] != 0:
            c_x = int(m["m10"] / m["m00"])
            c_y = int(m["m01"] / m["m00"])
            self.c_x = c_x
            self.c_y = c_y
            detected = True
        return FrameObjectWithDetectedCenterOfMass(input_object.get_frame(), input_object.camera_index, self.c_x,
                                                   self.c_y, detected)


class ShowCentersOfMass(mtl.OperationParent):
    def __init__(self):
        super().__init__()
        self.c_x = 0
        self.c_y = 0

    def run(self, input_object: FrameObjectWithDetectedCenterOfMass):
        c_x, c_y = input_object.get_center_of_mass()
        if input_object.was_detected:
            self.c_x, self.c_y = c_x, c_y
        return FrameObjectWithDetectedCenterOfMass(
            cv2.circle(cv2.cvtColor(input_object.get_frame(), cv2.COLOR_GRAY2BGR),
                       (c_x, c_y), 5, (0, 0, 255), -1),
            input_object.camera_index, c_x,
            c_y, input_object.was_detected)
