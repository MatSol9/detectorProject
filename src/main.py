from queue import Queue
from flask import Flask, jsonify, request

import camera_io.cameraIO as cameraIO
import multi_thread_lib.multiThreadLib as mtl
from data_model.dataModel import Config


app = Flask(__name__)
cameras = cameraIO.AllCameras()


def main():
    config = Config()
    data_from_cameras = []
    for index in config.get_camera_indexes():
        detection_minimums = config.get_detection_minimums(index)
        detection_maximums = config.get_detection_maximums(index)
        data_from_cameras.append(Queue())
        cameras.add_camera(index, 30, data_from_cameras, detection_minimums, detection_maximums)
    data_display = mtl.DataSink(data_from_cameras, cameraIO.CameraDisplay("Video"))
    # cameras.start_all_cameras()
    data_display.start()
    app.run()


def get_cameras():
    result = cameras.cameras_to_dict()
    return jsonify(result)


@app.route('/cameras', methods=['GET'])
def get_cameras_REST():
    return get_cameras()


@app.route('/cameras', methods=['PUT'])
def stop_start_cameras_REST():
    index: str = request.args.get("index")
    active: str = request.args.get("active")
    if active.__eq__("true"):
        cameras.start_camera(int(index))
        return "camera {} turned on".format(index), 200
    elif active.__eq__("false"):
        cameras.stop_camera(int(index))
        return "camera {} turned off".format(index), 200
    else:
        return 'bad request!', 400


if __name__ == "__main__":
    main()
