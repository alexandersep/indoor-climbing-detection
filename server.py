import os
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import uuid
from src.video_processing import process_video
from server_utils import *
from src.utils import *
from supabase import create_client
from supabase.lib.client_options import ClientOptions
from dotenv import load_dotenv
from flask_socketio import SocketIO

# Load environment variables from .env file
load_dotenv()

PROCESSED_VIDEOS_FOLDER_NAME = "processed_videos"
RAW_VIDEOS_FOLDER_NAME = "raw_videos"
TEMP_FOLDER_NAME = "temp"

# Create a Flask application instance
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins=['*', "http://localhost:8081"])
app.debug = True
CORS(app)  # Enable CORS for all sources

# Initialize the Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ROLE_KEY = os.environ.get("SUPABASE_ROLE_KEY")
ROOT_URL = os.environ.get("ROOT_URL")

supabase = create_client(
  SUPABASE_URL, 
  SUPABASE_ROLE_KEY,
  options=ClientOptions(
    auto_refresh_token=True,
    persist_session=True,
  )
)
# Define a route for video upload
@app.route('/vision-project/video-upload/<socketid>', methods=['POST'])
def video_upload(socketid):
    print("Request initiated with payload: " + str(request.files) + " socketid: " + str(socketid))
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files['video']
    original_file_path = os.path.join(RAW_VIDEOS_FOLDER_NAME, video.filename)

    # # if video.filename is alrady in storage
    if not os.path.exists(original_file_path):
        os.makedirs(RAW_VIDEOS_FOLDER_NAME, exist_ok=True)  # Ensure the directory exists
        video.save(original_file_path)

    response = supabase.table("processed_videos").select("*").eq("owner", video.filename).execute()
    print("supabase response: " + str(response))
    if len(response.data) > 0:
        return jsonify({"message": "Video already processed", "result": parse_video_data(response.data)})
    else:
        try:
            processed_outputs_path = os.path.join(PROCESSED_VIDEOS_FOLDER_NAME)
            if not os.path.exists(processed_outputs_path):
                os.makedirs(PROCESSED_VIDEOS_FOLDER_NAME, exist_ok=True)
            conversion_temp_path = os.path.join(TEMP_FOLDER_NAME, video.filename)
            if not os.path.exists(conversion_temp_path):
                os.makedirs(TEMP_FOLDER_NAME, exist_ok=True)

            # convert_with_moviepy(input_path=original_file_path, output_path=conversion_temp_path)

            results = process_video(original_file_path, processed_outputs_path, (socketio, socketid))
            print("Processing results:" + str(results))

            for vid in results:
                output_vid_name = vid[0]
                output_vid_events = vid[1]
                response = (
                    supabase.table("processed_videos")
                    .insert({"videoUrl": ROOT_URL + "/vision-project/get-video/" + output_vid_name, 
                             "owner": video.filename, 
                             "events": output_vid_events})
                    .execute()
                )

            # os.remove(conversion_temp_path);
            return jsonify({"message": "Video processed successfully", "result": parse_video_data(response.data)})
        
        except Exception as e:
            os.remove(original_file_path)
            return jsonify({"error": str(e)}), 500
        

@app.route('/vision-project/get-video/<filename>', methods=['GET'])
def get_video(filename):
    try:
        # Ensure the file exists in the directory
        if not os.path.exists(os.path.join(PROCESSED_VIDEOS_FOLDER_NAME, filename)):
            return jsonify({"error": "File not found"}), 404

        # Serve the file from the specified directory
        return send_from_directory(directory=PROCESSED_VIDEOS_FOLDER_NAME,
                                   path=filename,
                                   mimetype='video/mp4',
                                   as_attachment=False
                                   )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the server
if __name__ == '__main__':
    app.run(debug=True)
