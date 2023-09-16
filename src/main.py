from flask import Flask, jsonify, request

import cv2
import threading

import multi_thread_data_processing.multiThreadDataProcessing as mtl
from data_model.dataModel import Config
from camera_io.cameraIO import CameraDisplayPersonDetections
from camera_io.cameraIO import AllCameras

app = Flask(__name__)
cameras = AllCameras()


# def main():
#     data_display = mtl.DataSink(cameras.data_output, cameraIO.CameraDisplay("Video", cameras.camera_data))
#     data_display.start()
#     app.run()


# output_map = {}


def main():
    data_display = mtl.DataSink(cameras.data_output, CameraDisplayPersonDetections())
    data_display.start()
    app.run(host='0.0.0.0', port=2137)


def get_cameras():
    result = cameras.cameras_to_dict()
    return jsonify(result)
#
#
# @app.route('/cameras/output', methods=['GET'])
# def get_output_rest():
#     return jsonify(output_map)


@app.route('/cameras/get', methods=['GET'])
def get_cameras_rest():
    return get_cameras()


@app.route('/cameras/activate', methods=['PUT'])
def stop_start_cameras_rest():
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


@app.route('/cameras/create', methods=['POST'])
def create_camera_rest():
    try:
        index: int = int(request.args.get("index"))
        fps: float = float(request.args.get("fps"))
        x: int = int(request.args.get("x"))
        y: int = int(request.args.get("y"))
        angle: float = float(request.args.get("angle"))
    except:
        return "wrong arguments", 400
    print("Creating camera with index {}, fps {}, and starting point {},{}".format(index, fps, x, y))
    try:
        cameras.add_camera(index, fps, x, y, angle)
    except:
        return "index {} does not exist".format(index), 500
    return "camera {} created".format(index), 200


@app.route('/cameras/update', methods=['PUT'])
def update_camera_rest():
    try:
        index: int = int(request.args.get("index"))
        fps: float = float(request.args.get("fps"))
        x: int = int(request.args.get("x"))
        y: int = int(request.args.get("y"))
        angle: float = float(request.args.get("angle"))
    except:
        return "wrong arguments", 400
    if index not in cameras.indexes:
        return "camera {} does not exist".format(index), 500
    lock = threading.Lock()
    lock.acquire(blocking=True, timeout=-1)
    camera = cameras.all_cameras.get(index)
    camera.fps = fps
    camera.data_getter.period = 1/fps
    camera.x = x
    camera.y = y
    camera.angle = angle
    cameras.camera_data[index] = x, y, angle, camera.resolution, camera.cals_display_points()
    lock.release()
    return "camera {} updated".format(index), 200


@app.route('/cameras/free', methods=['GET'])
def get_available_indexes_rest():
    config = Config()
    result = []
    for index in range(config.get_max_search_index()):
        if index not in cameras.indexes:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                result.append(index)
                cap.release()
    return jsonify({"available": result}), 200


@app.route('/cameras/resolutions', methods=['GET'])
def get_available_resolutions_rest():
    config = Config()
    result = {}
    for index in range(config.get_max_search_index()):
        if index not in cameras.indexes:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                result[index] = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
    return "available indexes and resolutions: {}".format(result), 200


if __name__ == "__main__":
    main()
