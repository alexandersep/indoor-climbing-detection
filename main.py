from src.video_processing import * 

video_path = "resources/videos/green-climb.mp4"
output_path = "resources/videos/output/"

if __name__ == "__main__":
    print(process_video(video_path, output_path, debug=True))
