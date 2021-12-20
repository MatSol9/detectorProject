from __future__ import annotations

from typing import List, Dict
from typing import Tuple

import apriltag
import cv2
import numpy as np

import src.multi_thread_data_processing.multiThreadDataProcessing as mtl
from src.camera_io.cameraIO import Settings
from src.data_model.dataModel import FrameObject
from src.data_model.dataModel import FrameObjectWithDetectedCenterOfMass


class DetectObjectsTransform(mtl.OperationParent):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    def run(self, input_object: List[FrameObject]) -> FrameObject:
        frame = input_object[0]
        frame_gr = cv2.cvtColor(frame.get_frame(), cv2.COLOR_BGR2GRAY)
        centers: Dict[int, Tuple[int, int]] = {}
        rots: Dict = {}
        # config = self.settings.config
        # objects: List[int] = self.settings.indexes
        detector = apriltag.Detector(self.settings.options)
        results = detector.detect(frame_gr)
        for detected in results:
            if detected.tag_id in self.settings.tags:
                (c_x, c_y) = (int(detected.center[0]), int(detected.center[1]))
                centers[self.settings.tags_index.get(detected.tag_id)] = c_x, c_y
                (ptA, ptB, ptC, ptD) = detected.corners
                pt_length = np.sqrt(((ptA[0] - ptD[0])*(ptA[0] - ptD[0])) + ((ptA[1] - ptD[1])*(ptA[1] - ptD[1])))
                val = np.arccos(float((ptA[1] - ptD[1]) / pt_length))
                if ptA[0] - ptD[0] < 0:
                    val += np.pi
                rots[self.settings.tags_index.get(detected.tag_id)] = val
        #
        # for index in objects:
        #     frame_detected = cv2.inRange(frame.get_frame(), self.settings.get_detection_minimums(index),
        #                                  self.settings.get_detection_maximums(index))
        #     frame_cleared = cv2.morphologyEx(frame_detected, cv2.MORPH_OPEN,
        #                                      cv2.getStructuringElement(*config.get_morphology_options()))
        #     frame_cleared = cv2.morphologyEx(frame_cleared, cv2.MORPH_CLOSE,
        #                                      cv2.getStructuringElement(*config.get_morphology_options()))
        #     m = cv2.moments(frame_cleared)
        #     if m["m00"] != 0:
        #         c_x = int(m["m10"] / m["m00"])
        #         c_y = int(m["m01"] / m["m00"])
        #         centers[index] = c_x, c_y
        #         rots[index] = 0
        return FrameObjectWithDetectedCenterOfMass(frame.get_frame(), frame.camera_index, centers, rots)


class ShowCentersOfMass(mtl.OperationParent):
    def __init__(self):
        super().__init__()

    def run(self, input_object: FrameObjectWithDetectedCenterOfMass) -> FrameObjectWithDetectedCenterOfMass:
        frame = input_object.get_frame()
        for c_x, c_y in input_object.centers.values():
            frame = cv2.circle(input_object.get_frame(), (c_x, c_y), 5, (0, 0, 255), -1)
        return FrameObjectWithDetectedCenterOfMass(frame, input_object.camera_index, input_object.centers,
                                                   input_object.rots)

