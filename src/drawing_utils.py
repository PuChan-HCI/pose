# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MediaPipe solution drawing utils."""

import math
from typing import List, Mapping, Optional, Tuple, Union

import cv2
import dataclasses
import matplotlib.pyplot as plt
import numpy as np

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2

# --- Pyqtgraphを追加
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets
from PyQt5 import Qt
from PyQt5 import QtGui

# Call below first
app=QtWidgets.QApplication([])

# 2D plot
# win2d = pg.GraphicsWindow(title="Webcam")
win2d = pg.GraphicsLayoutWidget(title="Webcam")
win2d.resize(1500,750)
win2d.setBackground('k')
win2d.setFixedSize(1500,750)
# win2d.showFullScreen()
p1 = pg.PlotWidget()

# Fix widget aspect ratio
p1.setAspectLocked(True)

# 3d plot
win = gl.GLViewWidget()
win.setBackgroundColor('k')
# win.resize(1000,1000)
scale = 5   # 座標を正規化されているため，可視化用に5倍拡大する

# get a layout
# layoutgb = pg.QtGui.QGridLayout()
layoutgb =  QtWidgets.QGridLayout()
win2d.setLayout(layoutgb)
layoutgb.addWidget(win, 0, 0)
layoutgb.addWidget(p1, 0, 1)
p1.sizeHint = lambda: pg.QtCore.QSize(100, 100)
win.sizeHint = lambda: pg.QtCore.QSize(100, 100)
win.setSizePolicy(p1.sizePolicy())

img = pg.ImageItem(border='w')
p1.addItem(img)
# ---

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = WHITE_COLOR
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px


def draw_detection(
    image: np.ndarray,
    detection: detection_pb2.Detection,
    keypoint_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
    bbox_drawing_spec: DrawingSpec = DrawingSpec()):
    """Draws the detction bounding box and keypoints on the image.

    Args:
        image: A three channel RGB image represented as numpy ndarray.
        detection: A detection proto message to be annotated on the image.
        keypoint_drawing_spec: A DrawingSpec object that specifies the keypoints'
        drawing settings such as color, line thickness, and circle radius.
        bbox_drawing_spec: A DrawingSpec object that specifies the bounding box's
        drawing settings such as color and line thickness.

    Raises:
        ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
        b) If the location data is not relative data.
    """
    if not detection.location_data:
        return
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape

    location = detection.location_data
    if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
        raise ValueError(
            'LocationData must be relative for this drawing funtion to work.')
    # Draws keypoints.
    for keypoint in location.relative_keypoints:
        keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                       image_cols, image_rows)
    cv2.circle(image, keypoint_px, keypoint_drawing_spec.circle_radius,
               keypoint_drawing_spec.color, keypoint_drawing_spec.thickness)
    # Draws bounding box if exists.
    if not location.HasField('relative_bounding_box'):
        return
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
        image_rows)
    cv2.rectangle(image, rect_start_point, rect_end_point,
                  bbox_drawing_spec.color, bbox_drawing_spec.thickness)

def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec,Mapping[int, DrawingSpec]] = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec,Mapping[Tuple[int, int],DrawingSpec]] = DrawingSpec()):
    """Draws the landmarks and the connections on the image.

    Args:
        image: A three channel RGB image represented as numpy ndarray.
        landmark_list: A normalized landmark list proto message to be annotated on
        the image.
        connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
        landmark_drawing_spec: Either a DrawingSpec object or a mapping from
        hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
        settings such as color, line thickness, and circle radius.
        If this argument is explicitly set to None, no landmarks will be drawn.
        connection_drawing_spec: Either a DrawingSpec object or a mapping from
        hand connections to the DrawingSpecs that specifies the
        connections' drawing settings such as color and line thickness.
        If this argument is explicitly set to None, no landmark connections will
        be drawn.

    Raises:
        ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
        b) If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                cv2.line(image, idx_to_coordinates[start_idx],
                         idx_to_coordinates[end_idx], drawing_spec.color,
                         drawing_spec.thickness)
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = landmark_drawing_spec[idx] if isinstance(
                landmark_drawing_spec, Mapping) else landmark_drawing_spec
        # White circle border
        circle_border_radius = max(drawing_spec.circle_radius + 1,
                                   int(drawing_spec.circle_radius * 1.2))
        cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                   drawing_spec.thickness)
        # Fill color into the circle
        cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                   drawing_spec.color, drawing_spec.thickness)

def draw_axis(
    image: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    focal_length: Tuple[float, float] = (1.0, 1.0),
    principal_point: Tuple[float, float] = (0.0, 0.0),
    axis_length: float = 0.1,
    axis_drawing_spec: DrawingSpec = DrawingSpec()):

    """Draws the 3D axis on the image.

    Args:
        image: A three channel RGB image represented as numpy ndarray.
        rotation: Rotation matrix from object to camera coordinate frame.
        translation: Translation vector from object to camera coordinate frame.
        focal_length: camera focal length along x and y directions.
        principal_point: camera principal point in x and y.
        axis_length: length of the axis in the drawing.
        axis_drawing_spec: A DrawingSpec object that specifies the xyz axis
        drawing settings such as line thickness.

    Raises:
        ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
    """
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    # Create axis points in camera coordinate frame.
    axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axis_cam = np.matmul(rotation, axis_length*axis_world.T).T + translation
    x = axis_cam[..., 0]
    y = axis_cam[..., 1]
    z = axis_cam[..., 2]
    # Project 3D points to NDC space.
    fx, fy = focal_length
    px, py = principal_point
    x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1., 1.)
    y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1., 1.)
    # Convert from NDC space to image space.
    x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
    y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
    # Draw xyz axis on the image.
    origin = (x_im[0], y_im[0])
    x_axis = (x_im[1], y_im[1])
    y_axis = (x_im[2], y_im[2])
    z_axis = (x_im[3], y_im[3])
    cv2.arrowedLine(image, origin, x_axis, RED_COLOR, axis_drawing_spec.thickness)
    cv2.arrowedLine(image, origin, y_axis, GREEN_COLOR,
                    axis_drawing_spec.thickness)
    cv2.arrowedLine(image, origin, z_axis, BLUE_COLOR,
                    axis_drawing_spec.thickness)

def _normalize_color(color):
    return tuple(v / 255. for v in color)

def plot_landmarks(image: np.ndarray,
                   landmark_list: landmark_pb2.NormalizedLandmarkList,
                   connections: Optional[List[Tuple[int, int]]] = None,
                   landmark_drawing_spec: DrawingSpec = DrawingSpec(
                       color=RED_COLOR, thickness=5),
                   connection_drawing_spec: DrawingSpec = DrawingSpec(
                       color=BLACK_COLOR, thickness=5),
                   elevation: int = 10,
                   azimuth: int = 10):
    """Plot the landmarks and the connections in matplotlib 3d.

    Args:
        landmark_list: A normalized landmark list proto message to be plotted.
        connections: A list of landmark index tuples that specifies how landmarks to
        be connected.
        landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color and line thickness.
        connection_drawing_spec: A DrawingSpec object that specifies the
        connections' drawing settings such as color and line thickness.
        elevation: The elevation from which to view the plot.
        azimuth: the azimuth angle to rotate the plot.
    Raises:
        ValueError: If any connetions contain invalid landmark index.
    """

    # plot using pyqtgraph
    # Draw grid ------------
    win.clear()
    gx = gl.GLGridItem()
    gx.rotate(90, 0, 1, 0)
    gx.translate(-10, 0, 0)
    win.addItem(gx)
    gy = gl.GLGridItem()
    gy.rotate(90, 1, 0, 0)
    gy.translate(0, -10, 0)
    win.addItem(gy)
    gz = gl.GLGridItem()
    gz.translate(0, 0, -10)
    win.addItem(gz)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_rotate_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    img_flip = cv2.flip(img_rotate_90_clockwise, 0)
    img.setImage(img_flip)
    # ----------------------

    if not landmark_list:
        return

    # --- Matplotlib用のものを止める
    # plt.figure(figsize=(10, 10))
    # ax = plt.axes(projection='3d')
    # ax.view_init(elev=elevation, azim=azimuth)
    # ---

    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < _PRESENCE_THRESHOLD)):
            continue

        # --- Matplotlib用のものを止める
        # Get each landmark in 3D coordinate and plot in 3D
        # ax.scatter3D(
        #     xs=[-landmark.z],
        #     ys=[landmark.x],
        #     zs=[-landmark.y],
        #     color=_normalize_color(landmark_drawing_spec.color[::-1]),
        #     linewidth=landmark_drawing_spec.thickness)
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

        # --- Pyqtgraph
        # --- 点データをプロット
        points = np.array([-landmark.z * 0.5 * scale,
                           -landmark.x * scale,
                           -landmark.y * scale])
        sp2 = gl.GLScatterPlotItem(pos=points, color=(1, 1, 1, 1), size=10)
        win.addItem(sp2)
        # ---

    # Relationship between landmarks (line connections)
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:

                # Get coordinate pairs (start and stop) & plot in 3D
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                ]

                # --- Matplotlib用のものを止める
                # ax.plot3D(
                #     xs=[landmark_pair[0][0], landmark_pair[1][0]],
                #     ys=[landmark_pair[0][1], landmark_pair[1][1]],
                #     zs=[landmark_pair[0][2], landmark_pair[1][2]],
                #     color=_normalize_color(connection_drawing_spec.color[::-1]),
                #     linewidth=connection_drawing_spec.thickness)

                # --- Pyqtgraph
                start = [landmark_pair[0][0] * 0.5 * scale,
                         -landmark_pair[0][1] * scale,
                         landmark_pair[0][2] * scale]
                stop = [landmark_pair[1][0] * 0.5 * scale,
                        -landmark_pair[1][1] * scale,
                        landmark_pair[1][2] * scale]
                lines = gl.GLLinePlotItem(pos=np.array([start, stop]), color=(0, 1, 0, 1), width=1, mode='lines')
                win.addItem(lines)
                # ----------------------------------------------------

        # --- Matplotlib用のものを止める
        # plt.show()
        win.show()
        # pg.QtGui.QApplication.processEvents()
        # Qt.QApplication.processEvents()
        # app.processEvents()
        pg.QtWidgets.QApplication.processEvents()

