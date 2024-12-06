import datetime
import sys
import threading
import time
import yaml
from flask import Response
from flask import Flask
from flask import render_template
import pathlib

import cv2
import cv2.aruco as arc
import numpy as np
import picamera2

# local imports
import aruco_detection.utils.chess_calibration as calibration
from aruco_detection.utils.ip import get_ip_address

V3_CALIB_PATH = './aruco_detection/calibrations/calibration_v3_chess.yaml'
V2_CALIB_PATH = './aruco_detection/calibrations/calibration_v2_chess.yaml'
V1_CALIB_PATH = './aruco_detection/calibrations/calibration_v1_chess.yaml'


class ArucoDetector:

    calibrate:          bool
    streaming:          bool
    auto_measurements:  bool
    version:            str

    dictionary:         arc.Dictionary
    board:              arc.GridBoard
    detector_params:    arc.DetectorParameters
    detector:           arc.ArucoDetector

    picam:              picamera2.Picamera2
    picam_config:       picamera2.configuration
    orig_frame:         np.array
    out_frame:          bytes
    stream_if:          str

    cam_matrix:         np.array
    cam_dist_matrix:    np.array
    cam_dist_coeff:     np.array
    cam_roi:            np.array

    collect_img_task:   threading.Thread
    measurement_task:   threading.Thread
    calibration_task:   threading.Thread
    flask_task:         threading.Thread

    map1: np.array
    map2: np.array

    def __init__(self, version="v3", stream_if="wlan0", calibrate=False, streaming=False, auto_measurements=False):

        # init program parameters
        self.calibrate = calibrate
        self.streaming = streaming
        self.auto_measurements = auto_measurements
        self.version = version

        # init Aruco Foo
        self.dictionary = arc.getPredefinedDictionary(arc.DICT_6X6_250)
        self.board = arc.GridBoard((2, 3), 0.075, 0.01, self.dictionary)
        self.detector_params = arc.DetectorParameters()
        self.detector = arc.ArucoDetector(self.dictionary, self.detector_params)

        # init Picam
        self.picam = picamera2.Picamera2()
        if version == "v2":
            self.picam_config = self.picam.create_video_configuration(raw={"size": (1640, 1232)}, main={"format": "RGB888", "size": (640, 480)}, buffer_count=5)
        elif version == "v3":
            self.picam_config = self.picam.create_video_configuration(raw={"size": (2304, 1296)}, main={"format": "RGB888", "size": (1280, 720)}, buffer_count=5)
        elif version == "v1":
            self.picam_config = self.picam.create_video_configuration(raw={"size": (2592, 1944)}, main={"format": "RGB888", "size": (640, 480)}, buffer_count=5)

        self.picam.configure(self.picam_config)
        self.picam.start()
        self.orig_frame = None
        self.out_frame = None
        self.stream_if = stream_if
        time.sleep(2.0)

        # init camera calibration parameters, if new calibration not needed
        if not calibrate:
            self.__init_calibration()

        # init tasks
        self.collect_img_task = threading.Thread(target=self.__get_frame)
        self.measurement_task = threading.Thread(target=self.__measurements)
        self.calibration_task = threading.Thread(target=self.__generate_calib_data)
        self.flask_task = threading.Thread(target=self.__flask)

    # init cv2 calibration parameters
    def __init_calibration(self, calib_path=""):
        if calib_path == "":
            if self.version == "v2":
                calib_path = V2_CALIB_PATH
            elif self.version == "v3":
                calib_path = V3_CALIB_PATH
            elif self.version == "v1":
                calib_path = V1_CALIB_PATH

        # Calibration
        loadeddict = None
        with open(calib_path) as f:
            loadeddict = yaml.safe_load(f)

        mtx = loadeddict.get("camera_matrix")
        dist = loadeddict.get("dist_coeff")
        img = self.picam.capture_array()
        mtx = np.array(mtx)
        dist = np.array(dist)
        h, w = img.shape[:2]
        self.cam_matrix = mtx
        self.cam_dist_matrix, self.cam_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        self.cam_dist_coeff = dist
        #if version == "V3":  # not working at the moment
        #    self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.cam_matrix, self.cam_dist_coeff,
        #                                                               np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        #                                                               self.cam_dist_matrix, (w, h), cv2.CV_16SC2)
        print("Successfully calibrated camera!", file=sys.stderr)

    # start Aruco Detector; if calibrate: start calibration
    def start(self):
        self.collect_img_task.start()
        if self.streaming:
            self.flask_task.start()
        if self.calibrate:
            self.calibration_task.start()
            self.calibration_task.join()
        if self.auto_measurements:
            self.measurement_task.start()

    # web stream of current image to host_ip:5000
    def __flask(self):
        app = Flask(__name__)
        ip = get_ip_address(self.stream_if)

        def send_frames():
            while True:
                if self.out_frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + self.out_frame + b'\r\n')
                time.sleep(0.1)

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/video_feed')
        def video_feed():
            return Response(send_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        app.run(debug=True, threaded=True, host=ip, use_reloader=False)

    # function to capture calibration images and build a calibration
    def __generate_calib_data(self):
        path = "./calib_data/"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        count = 0
        time.sleep(3)
        print("Started Data Generation, press Enter to Capture, write exit to continue with calibration build", file=sys.stderr)
        while True:
            if self.picam is not None:
                print("Press Enter to Capture", file=sys.stderr)
                msg = input()
                if msg == 'exit':
                    break
                name = path + str(count) + ".jpg"
                img = self.picam.capture_array()
                cv2.imwrite(name, img)
                print(f"wrote image {count}", file=sys.stderr)
                count += 1
            time.sleep(0.1)
        msg = input("Desired Calibration File name, empty for standard: ")
        msg = "calibration_" + self.version + "_" + msg + ".yaml"

        calib_path = calibration.main(filename=msg)
        self.__init_calibration(calib_path=calib_path)

    # capture frame and analyze for video output
    def __get_frame(self):
        while True:
            self.orig_frame = self.picam.capture_array()
            # normal detection
            frame_cpy = np.copy(self.orig_frame)
            marker_corners, marker_ids, rejected_candidates = self.detector.detectMarkers(self.orig_frame)
            frame_out = np.copy(self.orig_frame)
            if marker_ids is not [] and marker_ids is not None:
                frame_out = arc.drawDetectedMarkers(frame_cpy, marker_corners, marker_ids)
            else:
                if not self.calibrate:
                    print("No markers found (Vis)", file=sys.stderr)
            ret, buffer = cv2.imencode('.jpg', frame_out)
            self.out_frame = buffer.tobytes()
            time.sleep(0.1)

    def get_out_frame(self):
        return self.out_frame

    def measurement(self):
        """measure translation and rotation of all visible aruco markers;
            return: list of marker IDs, list of translation vecs, list of rotation vecs"""
        if self.orig_frame is not None:
            dist_rotation_vec = []
            dist_translation_vec = []

            #distorted image analysis
            marker_corners, marker_ids, rejected_candidates = self.detector.detectMarkers(self.orig_frame)
            if marker_ids is not [] and marker_ids is not None:
                dist_rotation_vec, dist_translation_vec, objpts = cv2.aruco.estimatePoseSingleMarkers(marker_corners, 0.08,
                                                                                            self.cam_matrix,
                                                                                            self.cam_dist_coeff)
                distance = np.sqrt(dist_translation_vec[0][0][0] ** 2 + dist_translation_vec[0][0][1] ** 2 + dist_translation_vec[0][0][2] ** 2)


            return marker_ids, dist_translation_vec, dist_rotation_vec
        else:
            return [], [], []
            
    # measure translation and rotation of current image
    def __measurements(self):
        """measure distorted and undistorted translation and rotation of current image and print to stdout; values only for one visible marker"""
        msr_cnt = 0
        print("time; dist_trans; dist_rot; undist_trans; undist_rot")
        while True:
            print("Input message to name measurement, press enter to measure", file=sys.stderr)
            msg = input()
            print("measurement",  msr_cnt, msg)
            msr_cnt += 1
            j = 0
            while j < 10:
                if self.orig_frame is not None:
                    dist_rotation_vec = []
                    dist_translation_vec = []
                    undist_rotation_vec = []
                    undist_translation_vec = []

                    #distorted image analysis
                    marker_corners, marker_ids, rejected_candidates = self.detector.detectMarkers(self.orig_frame)
                    j += 1
                    if marker_ids is not [] and marker_ids is not None:
                        dist_rotation_vec, dist_translation_vec, objpts = cv2.aruco.estimatePoseSingleMarkers(marker_corners, 0.08,
                                                                                                    self.cam_matrix,
                                                                                                    self.cam_dist_coeff)
                        distance = np.sqrt(dist_translation_vec[0][0][0] ** 2 + dist_translation_vec[0][0][1] ** 2 + dist_translation_vec[0][0][2] ** 2)

                        print("Distorted Distance:", distance, file=sys.stderr)
                    else:
                        print(f"No markers found in distorted image, Measurement {msr_cnt}, Iteration {j}", file=sys.stderr)

                    #undistorted image analysis
                    img_gray = cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2GRAY)
                    img_undist = cv2.undistort(img_gray, self.cam_matrix, self.cam_dist_coeff, None, self.cam_dist_matrix)
                    marker_corners, marker_ids, rejected_candidates = self.detector.detectMarkers(img_undist)
                    if marker_ids is not [] and marker_ids is not None:
                        undist_rotation_vec, undist_translation_vec, objpts = cv2.aruco.estimatePoseSingleMarkers(marker_corners, 0.08,
                                                                                                    self.cam_dist_matrix,
                                                                                                    self.cam_dist_coeff)
                        distance = np.sqrt(undist_translation_vec[0][0][0] ** 2 + undist_translation_vec[0][0][1] ** 2 + undist_translation_vec[0][0][2] ** 2)
                        print("Undistorted Distance:", distance, file=sys.stderr)
                    else:
                        print(f"No markers found in undistorted image, Measurement {msr_cnt}, Iteration {j}", file=sys.stderr)



                    current_time = datetime.datetime.now()
                    print(current_time, ";", dist_translation_vec, ";", dist_rotation_vec, ";", undist_translation_vec,
                          ";", undist_rotation_vec)
                    time.sleep(0.3)
            print("Measurement written!", file=sys.stderr)


if __name__ == '__main__':
    #select Version from ["v2", "v3"]
    arc_detector = ArucoDetector(version="v1", stream_if="usb0")
    arc_detector.start()

"""
#    print(distance, file=sys.stderr)
                        #for marker_id, marker_corner, i in zip(marker_ids, marker_corners, total_markers):
                        #    #obj_points, img_points = self.board.matchImagePoints(marker_corner, marker_id)
                        #    #ret, rotation_vec, translation_vec = cv2.solvePnP(obj_points, img_points, self.cam_dist_matrix, self.cam_dist_coeff)
                        #    #deprecated
                        #    rotation_vec, translation_vec, objpts = cv2.aruco.estimatePoseSingleMarkers(marker_corner, 0.08, self.cam_dist_matrix, self.cam_dist_coeff)

                        #    distance = np.sqrt(translation_vec[0][0][0] ** 2 + translation_vec[0][0][1] ** 2 + translation_vec[0][0][2] ** 2)
                        #    print(distance, file=sys.stderr)
"""