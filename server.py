import os
from flask import Flask, jsonify, send_from_directory, request, Response
from flask_cors import CORS
from server_utils import *
from src.utils import *
from supabase import create_client
from supabase.lib.client_options import ClientOptions
from dotenv import load_dotenv
import threading
from jobs.processing_jobs import background_video_processing
import json

# Load environment variables from .env file
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ROLE_KEY = os.environ.get("SUPABASE_ROLE_KEY")
ROOT_URL = os.environ.get("ROOT_URL")
NGNINX_PROXY_LOCATION = os.environ.get("NGNINX_PROXY_LOCATION")

PROCESSED_VIDEOS_FOLDER_NAME = "processed_videos"
RAW_VIDEOS_FOLDER_NAME = "raw_videos"
TEMP_FOLDER_NAME = "temp"

# Create a Flask application instance
app = Flask(__name__)
app.debug = True
CORS(app)  # Enable CORS for all sources

# Initialize the Supabase client
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_ROLE_KEY,
    options=ClientOptions(auto_refresh_token=True, persist_session=True),
)


@app.route(NGNINX_PROXY_LOCATION + "/video-upload", methods=["POST"])
def video_upload():
    printd("Request initiated with payload: " + str(request.files))

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]

    alreadyProcessedResponse = (
        supabase.table("processed_videos")
        .select("*")
        .eq("owner", video.filename)
        .execute()
    )
    if alreadyProcessedResponse.data and len(alreadyProcessedResponse.data) > 0:
        return jsonify(
            {
                "message": "Video has already been processed",
                "already_processed": True,
                "data": parse_video_data(alreadyProcessedResponse.data),
            }
        )

    original_file_path = os.path.join(RAW_VIDEOS_FOLDER_NAME, video.filename)

    # If video doesn't exist locally, save it
    if not os.path.exists(original_file_path):
        os.makedirs(RAW_VIDEOS_FOLDER_NAME, exist_ok=True)
        video.save(original_file_path)

    insert_response = (
        supabase.table("jobs")
        .insert(
            {
                "status": "processing",
                "owner": video.filename,
            }
        )
        .execute()
    )
    job_id = insert_response.data[0]["job_id"]

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
        ),
        daemon=True,
    )
    worker_thread.start()

    return (
        jsonify(
            {
                "message": "Job created and processing started.",
                "job_id": job_id,
                "thread_id": worker_thread.native_id,
                "already_processed": False,
                "data": None,
            }
        ),
        202,
    )


@app.route(NGNINX_PROXY_LOCATION + "/get-job/<job_id>", methods=["GET"])
def get_job(job_id):
    try:
        response = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
        return jsonify(response.data[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route(NGNINX_PROXY_LOCATION + "/get-videos-from-owner", methods=["GET"])
def get_videos_from_owner():
    owner = request.args.get('owner')
    try:
        response = (
            supabase.table("processed_videos").select("*").eq("owner", owner).execute()
        )
        return jsonify(parse_video_data(response.data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
