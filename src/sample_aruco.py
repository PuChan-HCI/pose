#!/usr/bin/env python
# -*- coding: utf-8 -*
import cv2
import numpy as np
import math

aruco = cv2.aruco

# WEBカメラ
cap = cv2.VideoCapture(0)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# CORNER_REFINE_NONE, no refinement. CORNER_REFINE_SUBPIX, do subpixel refinement. CORNER_REFINE_CONTOUR use contour-Points
# parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

cameraMatrix = np.array([
            [472.93056928, 0, 339.01368175],
            [0, 470.63432376, 245.99105788],
            [0, 0, 1]], dtype='double',)
distCoeffs = np.array([[0.06497102], [- 0.73563667],  [0.01023954],  [0.01486544],  [1.44711414]])


cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def decimal_round(value, digit):
    value_multiply = value * 10 ** digit
    value_float    = value_multiply.astype(int)/( 10 ** digit)
    return value_float


def estimatePose(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    # trash = []
    # rvecs = []
    # tvecs = []
    i = 0
    for c in corners:
        # nada, R, t = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        # rvecs.append(R)
        # tvecs.append(t)
        # trash.append(nada)
        trash, rvecs, tvecs = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    return rvecs, tvecs, trash

def main():

    ret, frame = cap.read()

    # 変換処理ループ
    while ret == True:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary, parameters=parameters)
        #print(corners)
        #print(ids)
        #print(rejectedImgPoints)

        aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))

        for i, corner in enumerate( corners ):
            points = corner[0].astype(np.int32)
            cv2.polylines(frame, [points], True, (0,255,255))
            cv2.putText(frame, str(ids[i][0]), tuple(points[0]), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0), 1)

        if ids is not None:
            for i in range( ids.size ):
                # Calculate pose
                rvecs_new, tvecs_new, _objPoints = estimatePose(corners, 0.05, cameraMatrix, distCoeffs)
                (rotation_matrix, jacobian) = cv2.Rodrigues(rvecs_new)

                # Calculate 3D marker coordinates (m)
                marker_loc = np.zeros((3, 1), dtype=np.float64)
                marker_loc_world = rotation_matrix @ marker_loc + tvecs_new
                cv2.putText(frame, 'x : ' + str(decimal_round(marker_loc_world[0]*100,2)) + ' cm', (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame, 'y : ' + str(decimal_round(marker_loc_world[1]*100,2)) + ' cm', (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(frame, 'z : ' + str(decimal_round(marker_loc_world[2]*100,2)) + ' cm', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs_new, tvecs_new, 0.1)

        cv2.imshow('org', frame)

        # Escキーで終了
        key = cv2.waitKey(50)
        if key == 27: # ESC
            break

        # 次のフレーム読み込み
        ret, frame = cap.read()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass