from dotenv import load_dotenv
import os
from moviepy.editor import VideoFileClip

load_dotenv()

ROOT_URL = os.environ.get("ROOT_URL")

def parse_video_data(data):
    transformed_data = [
            {
                "videoUrl": video["videoUrl"],
                "created_at": video["created_at"],
                "id": video["id"],
                "owner": video["owner"],
                "events": [
                    {
                        "limb": event[0],
                        "hold": event[1],
                        "timestamp": float(event[2])
                    } for event in video["events"]
                ],
            } for video in data
        ]
    
    return transformed_data

def convert_with_moviepy(input_path: str, output_path: str) -> None:
    clip = VideoFileClip(input_path)
    clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        ffmpeg_params=['-crf','18', '-aspect', '9:16', '-s', '1080x1920']
    )
    clip.close()
