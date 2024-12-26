from dotenv import load_dotenv
import os
from moviepy.editor import VideoFileClip

load_dotenv()

ROOT_URL = os.environ.get("ROOT_URL")

def processed_video_data_to_labelled_json_object(data):
    transformed_data = [
            {
                "videoUrl": ROOT_URL + "/vision-project/get-video/" + video[0],
                "events": [
                    {
                        "limb": event[0],
                        "hold": event[1],
                        "timestamp": float(event[2])
                    } for event in video[1]
                ]
            } for video in data
        ]
    return transformed_data

def convert_with_moviepy(input_path: str, output_path: str) -> None:
    clip = VideoFileClip(input_path)
    clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=['-crf','18', '-aspect', '9:16']
    )
    clip.close()