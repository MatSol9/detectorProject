from queue import Queue

import camera_io.cameraIO as cameraIO
import image_transforms.imageTransforms as imageTransforms
import multi_thread_lib.multiThreadLib as mtl
from data_model.dataModel import Config


def main():
    config = Config()
    data_from_input = [Queue()]
    data_from_worker_detect_red = [Queue()]
    camera_index: int = config.get_camera_indexes()[0]
    data_getter = mtl.DataGetter(data_from_input, cameraIO.CameraReader(camera_index)).start()
    data_worker_detect_red = mtl.DataWorker(data_from_input, data_from_worker_detect_red, mtl.OperationChain()
                                            .add_operation(
        imageTransforms.DetectColoursThresholdsTransform(
            (config.get_detection_minimums(camera_index), config.get_detection_maximums(camera_index))))
                                            .add_operation(
        imageTransforms.MorphologyTransform())
                                            .add_operation(imageTransforms.GetMomentsTransform())
                                            .add_operation(imageTransforms.ShowCentersOfMass())).start()
    data_display = mtl.DataSink(data_from_worker_detect_red, cameraIO.CameraDisplay("Video")).start()


if __name__ == "__main__":
    main()
