from copy import deepcopy
from queue import Queue
from typing import Optional, List, Dict, Tuple

import apriltag
import cv2
import numpy as np

import src.image_transforms.imageTransforms as imageTransforms
import src.multi_thread_data_processing.multiThreadDataProcessing as mtl
from src.data_model.dataModel import Config
from src.data_model.dataModel import FrameObject
from src.data_model.dataModel import FrameObjectWithDetectedObjects


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
        self.first_frames = {}

    def sink_data(self, input_object: List[FrameObjectWithDetectedObjects]):
        frame_window = np.zeros((*self.window_size, 3))
        for detected_frame in input_object:
            if detected_frame.camera_index not in self.cameras:
                self.cameras.add(detected_frame.camera_index)
            self.first_frames[detected_frame.camera_index] = detected_frame.get_frame()
            self.detected_objects_centers[detected_frame.camera_index] = detected_frame.centers
            self.detected_objects_rots[detected_frame.camera_index] = detected_frame.rots
            for object_index in self.indexes:
                if object_index in self.detected_objects_centers[detected_frame.camera_index]:
                    x_from_camera = self.camera_data.get(detected_frame.camera_index)[0]
                    y_from_camera = self.camera_data.get(detected_frame.camera_index)[1]
                    cosine = np.cos(self.camera_data.get(detected_frame.camera_index)[2])
                    sine = np.sin(self.camera_data.get(detected_frame.camera_index)[2])
                    self.detected_objects_centers.get(detected_frame.camera_index)[object_index] = \
                        int(self.detected_objects_centers.get(detected_frame.camera_index).get(object_index)[0] + \
                        x_from_camera*cosine - y_from_camera*sine), \
                        int(self.detected_objects_centers.get(detected_frame.camera_index).get(object_index)[1] + \
                        x_from_camera*sine + y_from_camera*cosine)
        frames_to_display = deepcopy(self.first_frames)
        for object_index in self.indexes:
            i = 0
            x = 0
            y = 0
            rot = 0
            for camera_index in self.cameras:
                if object_index in self.detected_objects_centers.get(camera_index):
                    i += 1
                    x_p = self.detected_objects_centers.get(camera_index).get(object_index)[0]
                    x += x_p
                    y_p = self.detected_objects_centers.get(camera_index).get(object_index)[1]
                    y += y_p
                    rot_p = self.detected_objects_rots.get(camera_index).get(object_index)
                    rot += rot_p
                    x_from_camera = self.camera_data.get(camera_index)[0]
                    y_from_camera = self.camera_data.get(camera_index)[1]
                    cosine = np.cos(self.camera_data.get(camera_index)[2])
                    sine = np.sin(self.camera_data.get(camera_index)[2])
                    frames_to_display[camera_index] = cv2.circle(frames_to_display[camera_index], (
                        int(x_p - x_from_camera*cosine + y_from_camera*sine), int(y_p - x_from_camera*sine - y_from_camera*cosine)), 5,
                                                                 (255, 0, 0), -1)
                    frames_to_display[camera_index] = cv2.putText(frames_to_display[camera_index],
                                                                  "object: {}: rot: {}".format(object_index,
                                                                                               str(rot_p)), (
                                                                      int(x_p - x_from_camera*cosine + y_from_camera*sine),
                                                                      int(y_p - x_from_camera*sine - y_from_camera*cosine + 15)),
                                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if i != 0:
                x = x // i
                y = y // i
                rot = rot / i
                for camera_index in self.cameras:
                    x_from_camera = self.camera_data.get(camera_index)[0]
                    y_from_camera = self.camera_data.get(camera_index)[1]
                    cosine = np.cos(self.camera_data.get(camera_index)[2])
                    sine = np.sin(self.camera_data.get(camera_index)[2])
                    frames_to_display[camera_index] = cv2.putText(frames_to_display[camera_index],
                                                                  "object: {}: rot: {}".format(object_index, str(rot)),
                                                                  (int(x - x_from_camera*cosine + y_from_camera*sine),
                                                                   int(y - x_from_camera*sine - y_from_camera*cosine)),
                                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    frames_to_display[camera_index] = cv2.circle(frames_to_display[camera_index], (
                        int(x - x_from_camera*cosine + y_from_camera*sine), int(y - x_from_camera*sine - y_from_camera*cosine)), 5,
                                                                 (0, 0, 255), -1)
                cv2.putText(frame_window, "object: {}: rot: {}".format(object_index, str(rot)), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame_window, (x, y), 5, (0, 0, 255), -1)
        for camera_index in self.cameras:
            cv2.imshow("Camera: {}".format(camera_index), frames_to_display[camera_index])
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
        self.indexes: List[int] = list(temp_objects.keys())
        self.options = apriltag.DetectorOptions(families=self.config.get_tag_family())
        self.tags_index = {}
        self.tags = []
        for index in self.indexes:
            self.tags_index[temp_objects.get(index).get("tag_id")] = index
            self.tags.append(temp_objects.get(index).get("tag_id"))


class Camera:
    def __init__(self,
                 index: int,
                 fps: float,
                 x: int,
                 y: int,
                 data_output: List[Queue]):
        self.index = index
        self.fps = fps
        self.x = x
        self.y = y
        self.status = "INACTIVE"
        data_from_input = [Queue()]
        self.settings: Settings = Settings()
        self.data_getter = mtl.PeriodicDataGetter(data_from_input,
                                                  CameraReader(self.index),
                                                  self.fps)
        self.data_worker_detect = mtl.DataWorker(data_from_input,
                                                 data_output,
                                                 mtl.OperationChain()
                                                 .add_operation(
                                                     imageTransforms
                                                         .DetectObjectsTransform(
                                                         self.settings)))

    def start(self):
        self.data_getter.start()
        self.data_worker_detect.start()
        self.status = "ACTIVE"

    def stop(self):
        self.data_getter.stop()
        self.data_worker_detect.stop()
        self.status = "INACTIVE"

    def set_settings(self, settings: Settings):
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

    def add_camera(self, index: int, fps: float, x: int, y: int, angle: float):
        output_object: Queue = Queue()
        self.data_output.append(output_object)
        self.all_cameras[index] = Camera(index, fps, x, y, [output_object])
        self.indexes.append(index)
        self.camera_data[index] = x, y, angle

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
