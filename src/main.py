from queue import Queue
from flask import Flask, jsonify, request

import camera_io.cameraIO as cameraIO
import multi_thread_lib.multiThreadLib as mtl
from data_model.dataModel import Config


app = Flask(__name__)
cameras = []


def main():
    config = Config()
    data_from_cameras = []
    for index in config.get_camera_indexes():
        data_from_cameras.append(Queue())
        cameras.append(cameraIO.Camera(index, 30, data_from_cameras))
    data_display = mtl.DataSink(data_from_cameras, cameraIO.CameraDisplay("Video"))
    for camera in cameras:
        camera.start()
    data_display.start()
    app.run()


def get_cameras():
    result = {}
    for camera in cameras:
        result[camera.index] = camera.__str__()
    return jsonify(result)


@app.route('/cameras', methods=['GET'])
def camerasREST():
    return get_cameras()


if __name__ == "__main__":
    main()
