from queue import Queue

import camera_io.cameraIO as cameraIO
import multi_thread_lib.multiThreadLib as mtl
from data_model.dataModel import Config


def main():
    config = Config()
    data_from_worker_detect_red = [Queue()]
    camera_index: int = config.get_camera_indexes()[0]
    camera = cameraIO.Camera(camera_index, 30, data_from_worker_detect_red)
    data_display = mtl.DataSink(data_from_worker_detect_red, cameraIO.CameraDisplay("Video"))
    camera.start()
    data_display.start()


if __name__ == "__main__":
    main()
