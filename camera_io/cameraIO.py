import multi_thread_lib.multiThreadLib as mtl
import cv2


class CameraReader(mtl.GetParent):
    def __init__(self, camera_number: int):
        super(CameraReader, self).__init__()
        self.cap = cv2.VideoCapture(camera_number)

    def stop(self):
        self.cap.release()

    def __del__(self):
        self.cap.release()

    def get_data(self):
        ret, frame = self.cap.read()
        return frame


class CameraDisplay(mtl.SinkParent):
    def __init__(self, winname: str):
        super(CameraDisplay, self).__init__()
        self.winname = winname

    def sink_data(self, input_object: list):
        cv2.imshow(self.winname, input_object[0])

    def stop(self):
        cv2.destroyWindow(self.winname)

    def __del__(self):
        cv2.destroyWindow(self.winname)