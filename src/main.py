from queue import Queue
import multi_thread_lib.multiThreadLib as mtl
import camera_io.cameraIO as cameraIO
import image_transforms.imageTransforms as imageTransforms


def main():
    datamover = [Queue()]
    datamover1 = [Queue()]
    datagetter = mtl.DataGetter(datamover, cameraIO.CameraReader(0)).start()
    dataworker = mtl.DataWorker(datamover, datamover1, mtl.OperationChain()
                                .add_operation(
                                    imageTransforms.DetectColoursThresholdsTransform(((0, 0, 30), (30, 30, 60))))
                                .add_operation(
                                    imageTransforms.ClearDetectedTransform())).start()
    datagiver = mtl.DataSink(datamover1, cameraIO.CameraDisplay("Video")).start()


if __name__ == "__main__":
    main()
