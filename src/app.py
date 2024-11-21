import cv2 

from src.hold_processing import process_holds
from src.body_processing import *

video_path = "resources/videos/green-climb.mp4";

def app():
    video, frame_width, frame_height = load_video(video_path)
    
    mp_pose, pose, mp_drawing = pose_init(min_detection_confidence=0.5)

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        process_holds(frame)
        process_skeleton(frame, mp_pose, pose, mp_drawing)

        # Show the combined result
        cv2.namedWindow('Combined Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Combined Frame', int(frame_width/2), int(frame_height/2))
        cv2.imshow('Combined Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (video, frame_width, frame_height)
