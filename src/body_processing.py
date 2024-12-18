import cv2 
import mediapipe as mp
from src.constants.limb_names import *

def process_skeleton(frame, mp_pose, pose, mp_drawing):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    limb_list = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            limb_name = mp_pose.PoseLandmark(idx).name
            if limb_name in limb_names:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.putText(frame, limb_names[limb_name], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                limb_list.append( ([x, y], limb_name) )
    return limb_list

def pose_init(min_detection_confidence=0.5):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=min_detection_confidence)
    mp_drawing = mp.solutions.drawing_utils
    return mp_pose, pose, mp_drawing
