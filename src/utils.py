import cv2
import numpy as np

def dist(x1, y1, x2, y2):
    return np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

def load_video(video_path):
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    return video, frame_width, frame_height, codec, fps

def download_video_in_range(video_path, output_path, begin_frame, end_frame, left_hand_holds, right_hand_holds, left_foot_holds, right_foot_holds):
    """
    range: (begin_frame :: inclusive, end_frame :: inclusive)
    """
    video, frame_width, frame_height, codec, fps = load_video(video_path)
    if not video.isOpened():
        print(f"Failed to open video file: {video_path}")
        return

    output = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))
    video.set(cv2.CAP_PROP_POS_FRAMES, begin_frame)

    while begin_frame <= end_frame:
        success, frame = video.read()
        if not success:
            break

        # For debugging
        #for left_hand in left_hand_holds:
        #    ((x, y), (w, h), frameCount) = left_hand
        #    color = (255, 0, 0) # gbr
        #    cv2.putText(frame, "Left Hand: " + str(frameCount), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        #    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
        #for right_hand in right_hand_holds:
        #    ((x, y), (w, h), frameCount) = right_hand
        #    color = (0, 0, 255)
        #    cv2.putText(frame, "Right Hand: " + str(frameCount), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        #    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
        #for left_foot in left_foot_holds:
        #    ((x, y), (w, h), frameCount) = left_foot
        #    color = (125, 0, 0)
        #    cv2.putText(frame, "Left Foot: " + str(frameCount), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        #    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
        #for right_foot in right_foot_holds:
        #    ((x, y), (w, h), frameCount) = right_foot
        #    color = (0, 0, 125)
        #    cv2.putText(frame, "Right Foot: " + str(frameCount), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        #    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)

        output.write(frame)
        begin_frame += 1

    video.release()
    output.release()
