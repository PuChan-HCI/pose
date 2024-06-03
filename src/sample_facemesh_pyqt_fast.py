#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
import time
import math
from PyQt5 import QtWidgets
import pyqtgraph as pg
from collections import deque
from imutils.video import WebcamVideoStream

class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)
    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_faces", type=int, default=1)
    parser.add_argument('--refine_landmarks', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_faces = args.max_num_faces
    refine_landmarks = args.refine_landmarks
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # カメラ準備 ###############################################################
    # cap = cv.VideoCapture(cap_device)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap = WebcamVideoStream(src=0).start()

    # モデルロード #############################################################
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # GUI call below first
    app = QtWidgets.QApplication([])

    #  Graphic window
    win = pg.GraphicsLayoutWidget()
    win.setWindowTitle("Fast Plotting by PyQtGraph")
    win.resize(1500, 750)
    win.setBackground('k')

    # グラフ  ########################################################
    x = np.zeros(100)
    y = np.zeros(100)
    plot_line = pg.PlotWidget()
    plot_line.setYRange(0,1)
    plot_line.addItem(pg.PlotCurveItem(x, y))

    # Image
    plot_image = pg.PlotWidget()
    plot_image.setAspectLocked(True)

    # 時間計測開始
    time_sta = time.perf_counter()

    # Get the layout
    layoutgb = pg.QtWidgets.QGridLayout()
    win.setLayout(layoutgb)
    layoutgb.addWidget(plot_line, 0, 0)
    layoutgb.addWidget(plot_image, 0, 1)
    plot_image.sizeHint = lambda: pg.QtCore.QSize(500, 100)
    plot_line.sizeHint = lambda: pg.QtCore.QSize(500, 100)

    # Plot image
    img = pg.ImageItem(border='w')
    plot_image.addItem(img)


    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        # ret, image = cap.read()
        image = cap.read()
        # if not ret:
        #     break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        results = face_mesh.process(image)

        # 描画 ################################################################
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:

                # 虹彩の外接円の計算
                left_eye, right_eye = None, None
                if refine_landmarks:
                    left_eye, right_eye = calc_iris_min_enc_losingCircle(
                        debug_image,
                        face_landmarks,
                    )
                # 描画
                debug_image, delta = draw_landmarks(
                    debug_image,
                    face_landmarks,
                    refine_landmarks,
                    left_eye,
                    right_eye,
                )

                # 配列をキューと見たてて要素を追加・削除
                # plot_line.clear()
                x = np.append(x, ((time.perf_counter() - time_sta)))
                x = np.delete(x, 0)
                y = np.append(y, delta)
                y = np.delete(y, 0)
        else:
            x = np.append(x, ((time.perf_counter() - time_sta)))
            x = np.delete(x, 0)
            y = np.append(y, 0.0)
            y = np.delete(y, 0)

        # Draw
        plot_line.clear()       # Do this to avoid memory leak
        xmin = np.amin(x)
        plot_line.setXRange(xmin, xmin + 5)
        plot_line.addItem(pg.PlotCurveItem(x, y))
        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        img.setImage(cv.rotate(debug_image, cv.ROTATE_90_CLOCKWISE))

        # キー処理(ESC：終了) #################################################
        if cv.waitKey(5) & 0xFF == 27:
            break

        win.show()
        pg.QtWidgets.QApplication.processEvents()
    cap.release()


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_iris_min_enc_losingCircle(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append((landmark_x, landmark_y))

    left_eye_points = [
        landmark_point[468],
        landmark_point[469],
        landmark_point[470],
        landmark_point[471],
        landmark_point[472],
    ]
    right_eye_points = [
        landmark_point[473],
        landmark_point[474],
        landmark_point[475],
        landmark_point[476],
        landmark_point[477],
    ]

    left_eye_info = calc_min_enc_losingCircle(left_eye_points)
    right_eye_info = calc_min_enc_losingCircle(right_eye_points)

    return left_eye_info, right_eye_info


def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    return center, radius


def draw_landmarks(image, landmarks, refine_landmarks, left_eye, right_eye):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z
        # print(landmark_z)

        landmark_point.append((landmark_x, landmark_y))

        cv.circle(image, (landmark_x, landmark_y), 1, (0, 255, 0), 1)

    if len(landmark_point) > 0:
        # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

        # 左眉毛(55：内側、46：外側)
        cv.line(image, landmark_point[55], landmark_point[65], (0, 255, 0), 2)
        cv.line(image, landmark_point[65], landmark_point[52], (0, 255, 0), 2)
        cv.line(image, landmark_point[52], landmark_point[53], (0, 255, 0), 2)
        cv.line(image, landmark_point[53], landmark_point[46], (0, 255, 0), 2)

        # 右眉毛(285：内側、276：外側)
        cv.line(image, landmark_point[285], landmark_point[295], (0, 255, 0),
                2)
        cv.line(image, landmark_point[295], landmark_point[282], (0, 255, 0),
                2)
        cv.line(image, landmark_point[282], landmark_point[283], (0, 255, 0),
                2)
        cv.line(image, landmark_point[283], landmark_point[276], (0, 255, 0),
                2)

        # 左目 (133：目頭、246：目尻)
        cv.line(image, landmark_point[133], landmark_point[173], (0, 255, 0),
                2)
        cv.line(image, landmark_point[173], landmark_point[157], (0, 255, 0),
                2)
        cv.line(image, landmark_point[157], landmark_point[158], (0, 255, 0),
                2)
        cv.line(image, landmark_point[158], landmark_point[159], (0, 255, 0),
                2)
        cv.line(image, landmark_point[159], landmark_point[160], (0, 255, 0),
                2)
        cv.line(image, landmark_point[160], landmark_point[161], (0, 255, 0),
                2)
        cv.line(image, landmark_point[161], landmark_point[246], (0, 255, 0),
                2)

        cv.line(image, landmark_point[246], landmark_point[163], (0, 255, 0),
                2)
        cv.line(image, landmark_point[163], landmark_point[144], (0, 255, 0),
                2)
        cv.line(image, landmark_point[144], landmark_point[145], (0, 255, 0),
                2)
        cv.line(image, landmark_point[145], landmark_point[153], (0, 255, 0),
                2)
        cv.line(image, landmark_point[153], landmark_point[154], (0, 255, 0),
                2)
        cv.line(image, landmark_point[154], landmark_point[155], (0, 255, 0),
                2)
        cv.line(image, landmark_point[155], landmark_point[133], (0, 255, 0),
                2)

        # 右目 (362：目頭、466：目尻)
        cv.line(image, landmark_point[362], landmark_point[398], (0, 255, 0),
                2)
        cv.line(image, landmark_point[398], landmark_point[384], (0, 255, 0),
                2)
        cv.line(image, landmark_point[384], landmark_point[385], (0, 255, 0),
                2)
        cv.line(image, landmark_point[385], landmark_point[386], (0, 255, 0),
                2)
        cv.line(image, landmark_point[386], landmark_point[387], (0, 255, 0),
                2)
        cv.line(image, landmark_point[387], landmark_point[388], (0, 255, 0),
                2)
        cv.line(image, landmark_point[388], landmark_point[466], (0, 255, 0),
                2)

        cv.line(image, landmark_point[466], landmark_point[390], (0, 255, 0),
                2)
        cv.line(image, landmark_point[390], landmark_point[373], (0, 255, 0),
                2)
        cv.line(image, landmark_point[373], landmark_point[374], (0, 255, 0),
                2)
        cv.line(image, landmark_point[374], landmark_point[380], (0, 255, 0),
                2)
        cv.line(image, landmark_point[380], landmark_point[381], (0, 255, 0),
                2)
        cv.line(image, landmark_point[381], landmark_point[382], (0, 255, 0),
                2)
        cv.line(image, landmark_point[382], landmark_point[362], (0, 255, 0),
                2)

        # 口 (308：右端、78：左端)
        cv.line(image, landmark_point[308], landmark_point[415], (0, 255, 0),
                2)
        cv.line(image, landmark_point[415], landmark_point[310], (0, 255, 0),
                2)
        cv.line(image, landmark_point[310], landmark_point[311], (0, 255, 0),
                2)
        cv.line(image, landmark_point[311], landmark_point[312], (0, 255, 0),
                2)
        cv.line(image, landmark_point[312], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[82], (0, 255, 0), 2)
        cv.line(image, landmark_point[82], landmark_point[81], (0, 255, 0), 2)
        cv.line(image, landmark_point[81], landmark_point[80], (0, 255, 0), 2)
        cv.line(image, landmark_point[80], landmark_point[191], (0, 255, 0), 2)
        cv.line(image, landmark_point[191], landmark_point[78], (0, 255, 0), 2)

        cv.line(image, landmark_point[78], landmark_point[95], (0, 255, 0), 2)
        cv.line(image, landmark_point[95], landmark_point[88], (0, 255, 0), 2)
        cv.line(image, landmark_point[88], landmark_point[178], (0, 255, 0), 2)
        cv.line(image, landmark_point[178], landmark_point[87], (0, 255, 0), 2)
        cv.line(image, landmark_point[87], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[317], (0, 255, 0), 2)
        cv.line(image, landmark_point[317], landmark_point[402], (0, 255, 0),
                2)
        cv.line(image, landmark_point[402], landmark_point[318], (0, 255, 0),
                2)
        cv.line(image, landmark_point[318], landmark_point[324], (0, 255, 0),
                2)
        cv.line(image, landmark_point[324], landmark_point[308], (0, 255, 0),
                2)
        cv.circle(image,landmark_point[14],5,(0, 255, 0), 2)
        cv.circle(image, landmark_point[78], 5, (0, 255, 0), 2)
        cv.circle(image, landmark_point[13], 5, (0, 255, 0), 2)
        cv.circle(image, landmark_point[308], 5, (0, 255, 0), 2)

        dx = math.sqrt(pow(landmark_point[78][0]-landmark_point[308][0],2) + pow(landmark_point[78][1]-landmark_point[308][1],2))
        dy = math.sqrt(pow(landmark_point[13][0]-landmark_point[14][0],2) + pow(landmark_point[13][1] - landmark_point[14][1],2))
        delta = float(dy/dx)
        # print(dx, ", ", dy, " = ", delta)

        if refine_landmarks:
            # 虹彩：外接円
            cv.circle(image, left_eye[0], left_eye[1], (0, 255, 0), 2)
            cv.circle(image, right_eye[0], right_eye[1], (0, 255, 0), 2)

            # 左目：中心
            cv.circle(image, landmark_point[468], 2, (0, 0, 255), -1)
            # 左目：目頭側
            cv.circle(image, landmark_point[469], 2, (0, 0, 255), -1)
            # 左目：上側
            cv.circle(image, landmark_point[470], 2, (0, 0, 255), -1)
            # 左目：目尻側
            cv.circle(image, landmark_point[471], 2, (0, 0, 255), -1)
            # 左目：下側
            cv.circle(image, landmark_point[472], 2, (0, 0, 255), -1)
            # 右目：中心
            cv.circle(image, landmark_point[473], 2, (0, 0, 255), -1)
            # 右目：目尻側
            cv.circle(image, landmark_point[474], 2, (0, 0, 255), -1)
            # 右目：上側
            cv.circle(image, landmark_point[475], 2, (0, 0, 255), -1)
            # 右目：目頭側
            cv.circle(image, landmark_point[476], 2, (0, 0, 255), -1)
            # 右目：下側
            cv.circle(image, landmark_point[477], 2, (0, 0, 255), -1)

    return image, delta

if __name__ == '__main__':
    main()
    pg.exec()
