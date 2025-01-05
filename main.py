from src.video_processing import * 
from src.utils import *

video_path = "test.mp4"
output_path = "processed_videos"

if __name__ == "__main__":
    process_video(video_path, output_path, None)
    # convert_with_moviepy(os.path.join("processed_videos", "7c8fc9d7-a1f3-4057-b36b-f74d60ce1fdf.mp4"), "processed_videos/test.mp4")
