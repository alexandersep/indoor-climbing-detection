import cv2
import os
from src.hold_processing import *
from src.body_processing import *
from src.opticalflow_processing import *
from src.utils import *
import uuid
import sys

output_path_select_frame = "resources/videos/output/green-climb-trimmed-select.mp4"

def process_video(video_path, output_path, jobs_api):
    video, frame_width, frame_height, _, fps = load_video(video_path)
    if not video.isOpened():
        printd(f"Failed to open video file: {video_path}")
        return

    mp_pose, pose, mp_drawing = pose_init(min_detection_confidence=0.5)

    currentFrameCount = 0
    frameCount = 0
    started = False
    finished = False
    beginFrame = 0
    endFrame = 0

    prevStarted = False
    prevFinished = False

    first_frame_contour_bounding_boxes = []
    result = []
    labels = []
    left_hand_holds = []
    right_hand_holds = []
    left_foot_holds = []
    right_foot_holds = []

    ANNOTATED_VIDEO_PATH = "temp/" + str(uuid.uuid4()) + ".mp4";
    TEMP_FOLDER_NAME = "temp"

    if not os.path.exists(TEMP_FOLDER_NAME):
        os.makedirs(TEMP_FOLDER_NAME, exist_ok=True)

    video_writer = setup_video_writer(video, ANNOTATED_VIDEO_PATH)
    processed_video_render_data = []
    
    last_submitted_progress_update = 0
    contours, start_hold, end_hold = None, None, None

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frameCount == 0:
            contours, start_hold, end_hold = process_holds(frame)
            for contour in contours:
                cx, cy, cw, ch = cv2.boundingRect(contour)
                first_frame_contour_bounding_boxes.append((cx, cy, cw, ch))

            first_frame_contour_bounding_boxes = sorted(first_frame_contour_bounding_boxes, key=lambda x: x[1], reverse=True)

        limb_list = process_skeleton(frame, mp_pose, pose, mp_drawing)
        frame, isStarted, isFinished, left_hand_hold, right_hand_hold, left_foot_hold, right_foot_hold = process_hands(frame, limb_list, contours, start_hold, end_hold)
        frameCount += 1

        # Update loading bar

        progress_percentage = (frameCount / int(video.get(cv2.CAP_PROP_FRAME_COUNT))) * 100
        if jobs_api:
            supabase, job_id = jobs_api
            if round(progress_percentage) % 5 == 0 and last_submitted_progress_update != round(progress_percentage):
                supabase.table("jobs").update({ "processing_progress": round(progress_percentage)}).eq("job_id", job_id).execute()
                last_submitted_progress_update = round(progress_percentage)
        else:
            loading_bar = f"[{int(progress_percentage/100 * 50) * '='}{(50 - int(progress_percentage/100 * 50)) * ' '}] {progress_percentage/100 * 100:.2f}%"
            sys.stdout.write(f"\rProcessing video: {loading_bar}")
            sys.stdout.flush()
        # printd("isStarted", isStarted)
        # printd("isFinished", isFinished)

        if started and not finished:
            currentFrameCount += 1
            skip_holds(left_hand_holds, left_hand_hold, currentFrameCount, first_frame_contour_bounding_boxes, "Left Hand", labels, fps, start_hold, end_hold)
            skip_holds(right_hand_holds, right_hand_hold, currentFrameCount, first_frame_contour_bounding_boxes, "Right Hand", labels, fps, start_hold, end_hold)
            skip_holds(left_foot_holds, left_foot_hold, currentFrameCount, first_frame_contour_bounding_boxes, "Left Foot", labels, fps, start_hold, end_hold)
            skip_holds(right_foot_holds, right_foot_hold, currentFrameCount, first_frame_contour_bounding_boxes, "Right Foot", labels, fps, start_hold, end_hold)

        for hold_number, boundng_box in enumerate(first_frame_contour_bounding_boxes):
            x, y, w, h = boundng_box
            (sx, sy) = start_hold
            if (sx - 50 <= x <= sx + 50) and (sy - 50 <= y <= sy + 50):
                cv2.putText(frame, "Start Hold", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                continue

            (ex, ey) = end_hold
            if (ex - 50 <= x <= ex + 50) and (ey - 50 <= y <= ey + 50):
                cv2.putText(frame, "End Hold", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                continue

            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.putText(frame, "Hold " + str(hold_number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        isStartedLeft, isStartedRight = isStarted
        isFinishedLeft, isFinishedRight = isFinished
        if isStartedLeft and isStartedRight:
            started = True
            currentFrameCount = 0
        if isFinishedLeft and isFinishedRight:
            finished = True

        # printd("started", started)
        # printd("finished", finished)
        # printd("=================================================================")

        # Show the combined result
        # cv2.namedWindow('Combined Frame', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Combined Frame', int(frame_width/2), int(frame_height/2))
        # cv2.imshow('Combined Frame', frame)
        video_writer.write(frame)

        if started and prevStarted:
            beginFrame = frameCount
            currentFrameCount = 0
        if finished and prevFinished:
            endFrame = frameCount
            endFrame = frameCount
        prevStarted = isStartedLeft and isStartedRight
        prevFinished = isFinishedLeft and isFinishedRight

        if started and (finished and not prevFinished):
            distinct_file_name = str(uuid.uuid4()) + ".mp4"
            # endFrame += 180
            processed_video_render_data.append((distinct_file_name, beginFrame, endFrame))
            started = False
            prevStarted = False
            finished = False
            prevFinished = False

            result.append( (distinct_file_name, labels) )
            labels = []
            left_hand_holds = []
            right_hand_holds = []
            left_foot_holds = []
            right_foot_holds = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    video_writer.release()

    for distinct_file_name, start, end in processed_video_render_data:
        distinct_output_path = output_path + "/" + distinct_file_name
        distinct_temp_path = TEMP_FOLDER_NAME + "/" + distinct_file_name
        download_video_in_range(ANNOTATED_VIDEO_PATH, distinct_temp_path, start, end)
        convert_with_moviepy(input_path=distinct_temp_path, output_path=distinct_output_path)
        if os.path.exists(distinct_temp_path):
            printd("Removing bad codec file: " + distinct_temp_path)
            os.remove(distinct_temp_path)
        

    if os.path.exists(ANNOTATED_VIDEO_PATH):
        printd("Removing processed file: " + ANNOTATED_VIDEO_PATH)
        os.remove(ANNOTATED_VIDEO_PATH)

    cv2.destroyAllWindows()

    printd("\nProcessing complete." + str(result))

    return result

def skip_holds(holds, hold, frameCount, first_frame_contour_bounding_boxes, limb_name, labels, fps, start_hold, end_hold):
    # printd("this is skip_holds 1")
    ((x, y), (_, _)) = hold
    if hold == ((-1, -1), (-1, -1)):
        return


    isSkip = False
    for limb in holds:
        ((limb_x, limb_y), _, _) = limb
        distance = dist(x, y, limb_x, limb_y)

        if distance <= 100:
            isSkip = True

    if isSkip:
        return

    # printd("this is skip_holds 2")
    for hold_number, contour in enumerate(first_frame_contour_bounding_boxes):
        cx, cy, cw, ch = contour
        if (cx - 40 <= x <= cx + 40) and (cy - 40 <= y <= cy + 40):
            printd("other")
            holds.append( ((cx, cy), (cw, ch), frameCount) )
            (sx, sy) = start_hold
            if (sx - 40 <= cx <= sx + 40) and (sy - 40 <= cy <= sy + 40): # threshold so high because hold 7 is start hold and it already exists in list, not good
                printd("start")
                labels.append( (limb_name, "Start Hold", str(frameCount / fps)) )
                isSkip = True
                break

            (ex, ey) = end_hold
            if (ex - 40 <= cx <= ex + 40) and (ey - 40 <= cy <= ey + 40):
                printd("end")
                labels.append( (limb_name, "End Hold", str(frameCount / fps)) )
                isSkip = True
                break
            labels.append( (limb_name, "Hold " + str(hold_number), str(frameCount / fps)) )
            break
    # printd("this is skip_holds 3")

def setup_video_writer(video, filepath):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    return out
