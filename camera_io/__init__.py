from collections import deque
import multi_thread_lib.multiThreadLib as mtl
import cameraIO


datamover = [deque([])]
datagetter = mtl.DataGetter(datamover, cameraIO.CameraReader(0)).start()
# datagiver = mtl.DataSink(datamover, cameraIO.CameraDisplay("Webcam")).start()
