import datetime
import threading
from flask_cors import CORS
from src.video_processing import process_video
from server_utils import *
from src.utils import *


def background_video_processing(
    job_id,
    video_filename,
    original_file_path,
    processed_outputs_path,
    supabase,
    ROOT_URL,
    NGNINX_PROXY_LOCATION
):
    try:
        # 1. Long-running video processing
        results = process_video(
            original_file_path, processed_outputs_path, jobs_api=(supabase, job_id)
        )

        # 2. Insert processed results into `processed_videos`
        for vid in results:
            output_vid_name = vid[0]
            output_vid_events = vid[1]
            supabase.table("processed_videos").insert(
                {
                    "videoUrl": ROOT_URL
                    + NGNINX_PROXY_LOCATION
                    + "/get-video/"
                    + output_vid_name,
                    "owner": video_filename,
                    "events": output_vid_events,
                }
            ).execute()

        # 3. Mark the job as “completed” in the jobs table
        #    including a datetime stamp (e.g., UTC)
        supabase.table("jobs").update(
            {
                "status": "completed",
                "completed_at": datetime.datetime.now().isoformat(),  # or any format you like
            }
        ).eq("job_id", job_id).execute()

    except Exception as e:
        # If there's an error, mark as "failed" and store error message
        supabase.table("jobs").update(
            {
                "status": "failed",
                "message": str(e),
                "completed_at": datetime.datetime.now().isoformat(),
            }
        ).eq("job_id", job_id).execute()
