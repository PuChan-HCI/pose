import cv2
import mediapipe as mp
# from matplotlib import pyplot as plt

# --- 以下を追加・修正
import drawing_utils
mp_drawing = drawing_utils
# ---
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        h, w = image.shape[1], image.shape[0]

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.plot_landmarks(
            image,
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
cap.release()
# cv2.destroyAllWindows()