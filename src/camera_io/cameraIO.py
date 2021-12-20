from copy import deepcopy
from queue import Queue
from typing import Optional, List, Dict, Tuple, Any, Set

import apriltag
import cv2
import numpy as np

import src.image_transforms.imageTransforms as imageTransforms
import src.multi_thread_data_processing.multiThreadDataProcessing as mtl
from src.data_model.dataModel import Config
from src.data_model.dataModel import FrameObject
from src.data_model.dataModel import FrameObjectWithDetectedCenterOfMass


class CameraReader(mtl.GetParent):
    def __init__(self, camera_number: int):
        super(CameraReader, self).__init__()
        self.cap = cv2.VideoCapture(camera_number)
        if not self.cap.isOpened():
            raise Exception("Couldn't open camera {}".format(camera_number))
        self.index: int = camera_number

    def get_data(self) -> Optional[FrameObject]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return FrameObject(deepcopy(frame), self.index)


class CameraDisplay(mtl.SinkParent):
    def __init__(self, window_name: str, camera_data: Dict[int, Tuple[int, int]]):
        super(CameraDisplay, self).__init__()
        self.window_name = window_name
        config = Config()
        self.window_size: Tuple[int, int] = config.get_window_size()
        self.detected_objects_centers: Dict[int, Dict[int, Tuple[int, int]]] = {}
        self.detected_objects_rots: Dict[int, Dict[int, float]] = {}
        self.indexes = config.get_objects().keys()
        self.camera_data = camera_data
        self.cameras = set()

    def sink_data(self, input_object: List[FrameObjectWithDetectedCenterOfMass]):
        frame_window = np.zeros(self.window_size)
        for detected_frame in input_object:
            self.cameras.add(detected_frame.camera_index)
            self.detected_objects_centers[detected_frame.camera_index] = detected_frame.centers
            self.detected_objects_rots[detected_frame.camera_index] = detected_frame.rots
            for object_index in self.indexes:
                if object_index in self.detected_objects_centers[detected_frame.camera_index]:
                    self.detected_objects_centers.get(detected_frame.camera_index)[object_index] = self.detected_objects_centers.get(detected_frame.camera_index).get(object_index)[0] + self.camera_data.get(detected_frame.camera_index)[0], self.detected_objects_centers.get(detected_frame.camera_index).get(object_index)[1] + self.camera_data.get(detected_frame.camera_index)[1]
        for object_index in self.indexes:
            i = 0
            x = 0
            y = 0
            rot = 0
            for camera_index in self.cameras:
                if object_index in self.detected_objects_centers.get(camera_index):
                    i += 1
                    x += self.detected_objects_centers.get(camera_index).get(object_index)[0]
                    y += self.detected_objects_centers.get(camera_index).get(object_index)[1]
                    rot += self.detected_objects_rots.get(camera_index).get(object_index)
            if i != 0:
                x = x // i
                y = y // i
                rot = rot / i
                cv2.putText(frame_window, str(rot), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 2)
                cv2.circle(frame_window, (x, y), 5, 1, -1)
        cv2.imshow(self.window_name, frame_window)
        cv2.waitKey(1)

    def stop(self):
        cv2.destroyWindow(self.window_name)

    def __del__(self):
        cv2.destroyWindow(self.window_name)


class Settings:
    def __init__(self):
        self.config = Config()
        temp_objects = self.config.get_objects()
        self.objects: Dict[int, tuple] = {}
        self.indexes: List[int] = list(temp_objects.keys())
        for index in self.indexes:
            self.objects[index] = tuple(temp_objects.get(index).get("detection_minimums")), \
                                  tuple(temp_objects.get(index).get("detection_maximums"))
        self.options = apriltag.DetectorOptions(families=self.config.get_tag_family())
        self.tags_index = {}
        self.tags = []
        for index in self.indexes:
            self.tags_index[temp_objects.get(index).get("tag_id")] = index
            self.tags.append(temp_objects.get(index).get("tag_id"))

    def get_detection_minimums(self, index: int):
        return self.objects.get(index)[0]

    def get_detection_maximums(self, index: int):
        return self.objects.get(index)[1]


class Camera:
    def __init__(self, index: int, fps: float, x: int, y: int, data_output: List[Queue]):
        self.index = index
        self.fps = fps
        self.x = x
        self.y = y
        self.status = "INACTIVE"
        data_from_input = [Queue()]
        self.settings: Settings = Settings()
        self.data_getter = mtl.PeriodicDataGetter(data_from_input, CameraReader(self.index), self.fps)
        self.data_worker_detect = mtl.DataWorker(data_from_input, data_output, mtl.OperationChain()
                                                 .add_operation(imageTransforms.DetectObjectsTransform(self.settings))
                                                 # .add_operation(imageTransforms.ShowCentersOfMass())
                                                 )

    def start(self):
        self.data_getter.start()
        self.data_worker_detect.start()
        self.status = "ACTIVE"

    def stop(self):
        self.data_getter.stop()
        self.data_worker_detect.stop()
        self.status = "INACTIVE"

    def set_settings(self, settings: Dict[int, Dict[str, Tuple]]):
        self.settings = settings

    def to_dict(self):
        return {
            "fps": self.fps,
            "status": self.status,
            "handle point": (self.x, self.y)
        }

    def __str__(self) -> str:
        return " FPS: {}".format(self.fps)


class AllCameras:
    def __init__(self):
        self.all_cameras: Dict[int, Camera] = {}
        self.indexes: List[int] = []
        self.data_output: List[Queue] = []
        self.camera_data = {}

    def add_camera(self, index: int, fps: float, x: int, y: int):
        output_object: Queue = Queue()
        self.data_output.append(output_object)
        self.all_cameras[index] = Camera(index, fps, x, y, [output_object])
        self.indexes.append(index)
        self.camera_data[index] = x, y

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
