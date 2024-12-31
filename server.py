import os
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from src.video_processing import process_video
from server_utils import *
from src.utils import *
from supabase import create_client
from supabase.lib.client_options import ClientOptions
from dotenv import load_dotenv
from flask_socketio import SocketIO
import threading
from jobs.processing_jobs import *

# Load environment variables from .env file
load_dotenv()

PROCESSED_VIDEOS_FOLDER_NAME = "processed_videos"
RAW_VIDEOS_FOLDER_NAME = "raw_videos"
TEMP_FOLDER_NAME = "temp"

# Create a Flask application instance
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*") # https://climbing.oskarmroz.com
app.debug = True
CORS(app)  # Enable CORS for all sources

# Initialize the Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ROLE_KEY = os.environ.get("SUPABASE_ROLE_KEY")
ROOT_URL = os.environ.get("ROOT_URL")
NGNINX_PROXY_LOCATION = os.environ.get("NGNINX_PROXY_LOCATION")
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_ROLE_KEY,
    options=ClientOptions(
        auto_refresh_token=True,
        persist_session=True,
    ),
)


@app.route(NGNINX_PROXY_LOCATION + "/video-upload/<socketid>", methods=["POST"])
def video_upload(socketid):
    printd(
        "Request initiated with payload: "
        + str(request.files)
        + " socketid: "
        + str(socketid)
    )

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    original_file_path = os.path.join(RAW_VIDEOS_FOLDER_NAME, video.filename)

    # If video doesn't exist locally, save it
    if not os.path.exists(original_file_path):
        os.makedirs(RAW_VIDEOS_FOLDER_NAME, exist_ok=True)
        video.save(original_file_path)

    # 1. Create a job record in Supabase
    insert_response = (
        supabase.table("jobs")
        .insert(
            {
                "status": "processing",
                "owner": video.filename,
                # "completed_date": None  # If you want to be explicit, but can omit it
            }
        )
        .execute()
    )
    print(insert_response)
    job_id = insert_response.data[0]["job_id"]  # or your job primary key

    # 2. Spin up a background thread to do the processing
    processed_outputs_path = os.path.join(PROCESSED_VIDEOS_FOLDER_NAME)
    if not os.path.exists(processed_outputs_path):
        os.makedirs(PROCESSED_VIDEOS_FOLDER_NAME, exist_ok=True)

    worker_thread = threading.Thread(
        target=background_video_processing,
        args=(
            job_id,
            video.filename,
            original_file_path,
            processed_outputs_path,
            supabase,
            ROOT_URL,
            NGNINX_PROXY_LOCATION,
            socketio,
            socketid,
        ),
        daemon=True,
    )
    worker_thread.start()

    # 3. Return a response immediately
    return (
        jsonify({"message": "Job created and processing started.", "job_id": job_id}),
        202,
    )


@app.route(NGNINX_PROXY_LOCATION + "/get-video/<filename>", methods=["GET"])
def get_video(filename):
    try:
        # Ensure the file exists in the directory
        if not os.path.exists(os.path.join(PROCESSED_VIDEOS_FOLDER_NAME, filename)):
            return jsonify({"error": "File not found"}), 404

        # Serve the file from the specified directory
        return send_from_directory(
            directory=PROCESSED_VIDEOS_FOLDER_NAME,
            path=filename,
            mimetype="video/mp4",
            as_attachment=False,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route(NGNINX_PROXY_LOCATION + "/", methods=["GET"])
def get_test():
    return jsonify({"message": "Successfully Connected"})


# Start the server
if __name__ == "__main__":
    app.run(debug=True)
