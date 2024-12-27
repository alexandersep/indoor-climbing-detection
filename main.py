from src.video_processing import * 
from src.utils import *

video_path = "resources/videos/green-climb.mp4"
output_path = "resources/videos/output/"

if __name__ == "__main__":
    # print(process_video(video_path, output_path, debug=True))
    convert_with_moviepy(os.path.join("processed_videos", "7c8fc9d7-a1f3-4057-b36b-f74d60ce1fdf.mp4"), "processed_videos/test.mp4")
