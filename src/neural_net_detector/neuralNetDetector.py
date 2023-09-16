import time

import cv2.gapi
import torch.cuda

from typing import List, Tuple
from PIL import Image

import numpy as np


class Detector:
    def get_bboxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        pass


class MockedDetector(Detector):
    def get_bboxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return [(20, 40, 50, 100), (70, 90, 10, 120)]


class BaseYoloDetector(Detector):
    def __init__(self, model):
        self.__model = model
        self.__model.conf = 0.25  # NMS confidence threshold
        self.__model.iou = 0.45  # NMS IoU threshold
#        self.__model.cuda()
        self.__model.classes = [0]
        self.__model.agnostic = False  # NMS class-agnostic
        self.__model.multi_label = False  # NMS multiple labels per box
        self.__model.max_det = 5  # maximum number of detections per image

    def get_bboxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:

        results = self.__model(frame)
        results.print()
        predictions = results.pred[0]
        res = []
        if predictions.shape[0] == 0:
            return res
        for i in range(predictions.shape[0]):
            x1, y1, x2, y2 = predictions[i, :4]
            res.append((int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)))
        return res
