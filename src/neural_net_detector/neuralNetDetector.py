from typing import List, Tuple

import numpy as np


class Detector:
    def get_bboxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        pass


class MockedDetector(Detector):
    def get_bboxes(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return [(20, 40, 50, 100), (70, 90, 10, 120)]
