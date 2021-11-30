from copy import deepcopy

import cv2

import src.multi_thread_lib.multiThreadLib as mtl
from src.data_model.dataModel import FrameObject


class CameraReader(mtl.GetParent):
    def __init__(self, camera_number: int):
        super(CameraReader, self).__init__()
        self.cap = cv2.VideoCapture(camera_number)
        self.index: int = camera_number

    def get_data(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return FrameObject(deepcopy(frame), self.index)


class CameraDisplay(mtl.SinkParent):
    def __init__(self, window_name: str):
        super(CameraDisplay, self).__init__()
        self.window_name = window_name

    def sink_data(self, input_object: list):
        frame: FrameObject = deepcopy(input_object[0])
        cv2.imshow(self.window_name, frame.get_frame())
        cv2.waitKey(1)

    def stop(self):
        cv2.destroyWindow(self.window_name)

    def __del__(self):
        cv2.destroyWindow(self.window_name)
