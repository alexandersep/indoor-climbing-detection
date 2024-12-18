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

def download_video_in_range(video_path, output_path, begin_frame, end_frame):
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
        output.write(frame)
        begin_frame += 1

    video.release()
    output.release()
