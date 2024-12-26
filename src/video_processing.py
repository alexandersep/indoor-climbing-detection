import cv2
import os
import sys
from src.hold_processing import *
from src.body_processing import *
from src.opticalflow_processing import *
from src.utils import *
import uuid

output_path_select_frame = "resources/videos/output/green-climb-trimmed-select.mp4"

def process_video(video_path, output_path, debug):
    print("1")
    video, frame_width, frame_height, fps = load_video(video_path)
    if not video.isOpened():
        print(f"Failed to open video file: {video_path}")
        return

    print("2")
    mp_pose, pose, mp_drawing = pose_init(min_detection_confidence=0.5)
    print("3")

    frameCount = 0
    started = False
    finished = False
    beginFrame = 0
    endFrame = 0

    prevStarted = False
    prevFinished = False

    first_frame_contours = []
    isFirstFrame = True
    result = []
    labels = []
    left_hand_holds = []
    right_hand_holds = []
    left_foot_holds = []
    right_foot_holds = []

    PROCESSED_VIDEO_PATH = "resources/videos/output/" + str(uuid.uuid4()) + ".mp4";
    video_writer = setup_video_writer(video, PROCESSED_VIDEO_PATH)
    processed_video_render_data = []

    print("4")
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        contours, start_hold, end_hold = process_holds(frame)
        if isFirstFrame:
            isFirstFrame = False
            for contour in contours:
                cx, cy, cw, ch = cv2.boundingRect(contour)
                first_frame_contours.append((cx, cy, cw, ch))
            first_frame_contours = sorted(first_frame_contours, key=lambda x: x[1], reverse=True)
        limb_list = process_skeleton(frame, mp_pose, pose, mp_drawing)
        frame, isStarted, isFinished, left_hand_hold, right_hand_hold, left_foot_hold, right_foot_hold = process_hands(frame, limb_list, contours, start_hold, end_hold)
        frameCount += 1

        # Update loading bar
        #progress = frameCount / int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #loading_bar = f"[{int(progress * 50) * '='}{(50 - int(progress * 50)) * ' '}] {progress * 100:.2f}%"
        #sys.stdout.write(f"\rProcessing video: {loading_bar}")
        #sys.stdout.flush()
        print("isStarted", isStarted)
        print("isFinished", isFinished)

        if started and not finished:
            skip_holds(left_hand_holds, left_hand_hold, frameCount, first_frame_contours, "Left Hand", labels, fps)
            skip_holds(right_hand_holds, right_hand_hold, frameCount, first_frame_contours, "Right Hand", labels, fps)
            skip_holds(left_foot_holds, left_foot_hold, frameCount, first_frame_contours, "Left Foot", labels, fps)
            skip_holds(right_foot_holds, right_foot_hold, frameCount, first_frame_contours, "Right Foot", labels, fps)

        isStartedLeft, isStartedRight = isStarted
        isFinishedLeft, isFinishedRight = isFinished
        if isStartedLeft and isStartedRight:
            started = True
        if isFinishedLeft and isFinishedRight:
            finished = True

        print("started", started)
        print("finished", finished)
        print("=================================================================")

        # Show the combined result
        cv2.namedWindow('Combined Frame', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Combined Frame', int(frame_width/2), int(frame_height/2))
        cv2.imshow('Combined Frame', frame)
        video_writer.write(frame)

        if started and prevStarted:
            beginFrame = frameCount
        if finished and prevFinished:
            endFrame = frameCount
        prevStarted = isStartedLeft and isStartedRight
        prevFinished = isFinishedLeft and isFinishedRight

        if started and (finished and not prevFinished):
            distinct_file_name = str(uuid.uuid4()) + ".mp4"
            distinct_path = output_path + "/" + distinct_file_name;
            endFrame += 180
            processed_video_render_data.append((distinct_path, beginFrame, endFrame))
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

    for distinct_path, start, end in processed_video_render_data:
        download_video_in_range(PROCESSED_VIDEO_PATH, distinct_path, start, end)

    print("Removing processed file: " + PROCESSED_VIDEO_PATH)
    # Remove temporary processed file after splitting into separate files.
    if os.path.exists(PROCESSED_VIDEO_PATH):
        os.remove(PROCESSED_VIDEO_PATH)

    cv2.destroyAllWindows()

    print("\nProcessing complete.", result)

    return result

def skip_holds(holds, hold, frameCount, first_frame_contours, limb_name, labels, fps):
    print("this is skip_holds 1")
    ((x, y), (_, _)) = hold
    if hold == ((-1, -1), (-1, -1)):
        return

    isSkip = False
    for left_hand in holds:
        ((limb_x, limb_y), _, _) = left_hand
        distance = dist(x, y, limb_x, limb_y)
        if distance <= 100:
            isSkip = True
    print("this is skip_holds 2")
    if not isSkip:
        for hold_number, contour in enumerate(first_frame_contours):
            cx, cy, cw, ch = contour
            if (cx - 40 <= x <= cx + 40) and (cy - 40 <= y <= cy + 40):
                holds.append( ((cx, cy), (cw, ch), frameCount) )
                labels.append( (limb_name, "Hold " + str(hold_number), str(frameCount / fps)) )
                break
    print("this is skip_holds 3")

def setup_video_writer(video, filepath):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    return out
