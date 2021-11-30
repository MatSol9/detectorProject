from queue import Queue
import multi_thread_lib.multiThreadLib as mtl
import camera_io.cameraIO as cameraIO
import image_transforms.imageTransforms as imageTransforms


def main():
    data_from_input = [Queue()]
    data_from_worker_detect_red = [Queue()]
    data_getter = mtl.DataGetter(data_from_input, cameraIO.CameraReader(0)).start()
    data_worker_detect_red = mtl.DataWorker(data_from_input, data_from_worker_detect_red, mtl.OperationChain()
                                            .add_operation(
        imageTransforms.DetectColoursThresholdsTransform(((0, 0, 30), (30, 30, 60))))
                                            .add_operation(
        imageTransforms.ClearDetectedTransform())
                                            .add_operation(imageTransforms.GetMomentsTransform())
                                            .add_operation(imageTransforms.ShowCentersOfMass())).start()
    data_display = mtl.DataSink(data_from_worker_detect_red, cameraIO.CameraDisplay("Video")).start()


if __name__ == "__main__":
    main()
