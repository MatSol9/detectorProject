from copy import deepcopy
from queue import Queue
from typing import Optional, List, Dict

import cv2

import src.multi_thread_lib.multiThreadLib as mtl
from src.data_model.dataModel import FrameObject
from src.data_model.dataModel import Config
import src.image_transforms.imageTransforms as imageTransforms


class CameraReader(mtl.GetParent):
    def __init__(self, camera_number: int):
        super(CameraReader, self).__init__()
        self.cap = cv2.VideoCapture(camera_number)
        self.index: int = camera_number

    def get_data(self) -> Optional[FrameObject]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return FrameObject(deepcopy(frame), self.index)

    def stop(self):
        self.cap.release()


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


class Camera:
    def __init__(self, index: int, fps: float, data_output: List[Queue], detection_minimums, detection_maximums):
        self.index = index
        self.fps = fps
        self.status = "INACTIVE"
        data_from_input = [Queue()]
        self.config = Config()
        self.detection_minimums = detection_minimums
        self.detection_maximums = detection_maximums
        self.data_getter = mtl.PeriodicDataGetter(data_from_input, CameraReader(self.index), self.fps)
        self.data_worker_detect = mtl.DataWorker(data_from_input, data_output, mtl.OperationChain()
                                                 .add_operation(
            imageTransforms.DetectColoursThresholdsTransform(
                (self.detection_minimums, self.detection_maximums)))
                                                 .add_operation(
            imageTransforms.MorphologyTransform())
                                                 .add_operation(imageTransforms.GetMomentsTransform())
                                                 .add_operation(imageTransforms.ShowCentersOfMass()))

    def start(self):
        self.data_getter.start()
        self.data_worker_detect.start()
        self.status = "ACTIVE"

    def stop(self):
        self.data_getter.stop()
        self.data_worker_detect.stop()
        self.status = "INACTIVE"

    def to_dict(self):
        return {
            "fps": self.fps,
            "status": self.status,
            "detection_minimums": self.detection_minimums,
            "detection_maximums": self.detection_maximums
        }

    def __str__(self) -> str:
        return " FPS: {}".format(self.fps)


class AllCameras:
    def __init__(self):
        self.all_cameras: Dict[int, Camera] = {}
        self.indexes: List[int] = []

    def add_camera(self, index: int, fps: float, data_output: List[Queue], detection_minimums, detection_maximums):
        self.all_cameras[index] = Camera(index, fps, data_output, detection_minimums, detection_maximums)
        self.indexes.append(index)

    def start_camera(self, index: int):
        self.all_cameras[index].start()

    def stop_camera(self, index: int):
        self.all_cameras[index].stop()

    def start_all_cameras(self):
        for index in self.indexes:
            self.start_camera(index)

    def stop_all_cameras(self):
        for index in self.indexes:
            self.stop_camera(index)

    def cameras_to_dict(self) -> Dict:
        result = {}
        for index in self.indexes:
            result[index] = self.all_cameras[index].to_dict()
        return result

    def remove_camera(self, index: int):
        self.stop_camera(index)
        self.all_cameras.pop(index)
