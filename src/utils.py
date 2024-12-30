import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
import os

def dist(x1, y1, x2, y2):
    return np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

def load_video(video_path):
    printd(video_path)
    video = cv2.VideoCapture(video_path)
    printd(video)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    printd(frame_width)
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    printd(frame_height)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    printd(fps)
    
    return video, frame_width, frame_height, codec, fps

def download_video_in_range(video_path, output_path, begin_frame, end_frame):
    printd("Attempting to split the file at " + str(video_path))
    video, frame_width, frame_height, codec, fps = load_video(video_path)
    if not video.isOpened():
        printd(f"Failed to open video file: {video_path}")
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
    
    printd("Successfully split the file at " + str(video_path))
    printd("New file at " + str(output_path))
    
def convert_with_moviepy(input_path: str, output_path: str) -> None:
    printd("Attempting to convert with moviepy")
    clip = VideoFileClip(input_path)
    clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=['-crf','18', '-aspect', '9:16', '-s', '1080x1920']
    )
    clip.close()

def printd(message):
    load_dotenv()
    PROD = os.environ.get("PROD")
    if PROD != 'True':
        print("[debug] ", message)