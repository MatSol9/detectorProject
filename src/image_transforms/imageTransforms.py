from __future__ import annotations

from typing import List, Dict
from typing import Tuple

import apriltag
import cv2
import numpy as np

import src.multi_thread_data_processing.multiThreadDataProcessing as mtl
from src.camera_io.cameraIO import Settings
from src.data_model.dataModel import FrameObject
from src.data_model.dataModel import FrameObjectWithDetectedObjects
from src.data_model.dataModel import FrameObjectWithBoundingBoxes
from src.neural_net_detector.neuralNetDetector import Detector


class DetectObjectsTransform(mtl.OperationParent):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    def run(self, input_object: List[FrameObject]) -> FrameObject:
        frame = input_object[0]
        frame_gr = cv2.cvtColor(frame.get_frame(), cv2.COLOR_BGR2GRAY)
        centers: Dict[int, Tuple[int, int]] = {}
        rots: Dict = {}
        detector = apriltag.Detector(self.settings.options)
        results = detector.detect(frame_gr)
        for detected in results:
            if detected.tag_id in self.settings.tags:
                (c_x, c_y) = (int(detected.center[0]), int(detected.center[1]))
                centers[self.settings.tags_index.get(detected.tag_id)] = c_x, c_y
                (ptA, ptB, ptC, ptD) = detected.corners
                pt_length = np.sqrt(((ptA[0] - ptD[0])*(ptA[0] - ptD[0])) + ((ptA[1] - ptD[1])*(ptA[1] - ptD[1])))
                val = np.arccos(float((ptA[1] - ptD[1]) / pt_length))
                if ptA[0] - ptD[0] > 0:
                    val = 2 * np.pi - val
                rots[self.settings.tags_index.get(detected.tag_id)] = val
        return FrameObjectWithDetectedObjects(frame.get_frame(), frame.get_index(), centers, rots)


class DetectObjectWithModelTransform(mtl.OperationParent):
    def __init__(self, detector: Detector):
        super().__init__()
        self.__detector = detector

    def run(self, input_object: List[FrameObject]) -> FrameObject:
        frame = input_object[0]
        bboxes = self.__detector.get_bboxes(frame.get_frame())
        return FrameObjectWithBoundingBoxes(frame.get_frame(), frame.get_index(), bboxes)


class ShowCentersOfMass(mtl.OperationParent):
    def __init__(self):
        super().__init__()

    def run(self, input_object: FrameObjectWithDetectedObjects) -> FrameObjectWithDetectedObjects:
        frame = input_object.get_frame()
        for c_x, c_y in input_object.centers.values():
            frame = cv2.circle(input_object.get_frame(), (c_x, c_y), 5, (0, 0, 255), -1)
        return FrameObjectWithDetectedObjects(frame, input_object.get_index(), input_object.centers,
                                              input_object.rots)

