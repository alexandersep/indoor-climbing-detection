import cv2 

from src.hold_processing import *
from src.body_processing import *
from src.opticalflow_processing import *
from src.utils import *

import uuid

output_path_select_frame = "resources/videos/output/green-climb-trimmed-select.mp4"

def process_video(video_path, output_path):
    video, frame_width, frame_height, _, fps = load_video(video_path)
    if not video.isOpened():
        print(f"Failed to open video file: {video_path}")
        return
    
    mp_pose, pose, mp_drawing = pose_init(min_detection_confidence=0.5)

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

        # Show the combined result
        cv2.namedWindow('Combined Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Combined Frame', int(frame_width/2), int(frame_height/2))
        cv2.imshow('Combined Frame', frame)

        if started and prevStarted:
            beginFrame = frameCount
        if finished and prevFinished:
            endFrame = frameCount
        prevStarted = isStartedLeft and isStartedRight
        prevFinished = isFinishedLeft and isFinishedRight

        if started and (finished and not prevFinished):
            distinct_path = output_path + str(uuid.uuid4()) + ".mp4"
            endFrame += 90
            download_video_in_range(video_path, distinct_path, beginFrame, endFrame, left_hand_holds, right_hand_holds, left_foot_holds, right_foot_holds)
            started = False
            prevStarted = False
            finished = False
            prevFinished = False

            result.append( (distinct_path, labels) )
            labels = []
            left_hand_holds = []
            right_hand_holds = []
            left_foot_holds = []
            right_foot_holds = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    return result

def skip_holds(holds, hold, frameCount, first_frame_contours, limb_name, labels, fps):
    ((x, y), (_, _)) = hold
    if hold == ((-1, -1), (-1, -1)):
        return

    isSkip = False
    for left_hand in holds:
        ((limb_x, limb_y), _, _) = left_hand
        distance = dist(x, y, limb_x, limb_y)
        if distance <= 100:
            isSkip = True
    if not isSkip:
        for hold_number, contour in enumerate(first_frame_contours):
            cx, cy, cw, ch = contour
            if (cx - 40 <= x <= cx + 40) and (cy - 40 <= y <= cy + 40):
                holds.append( ((cx, cy), (cw, ch), frameCount) )
                labels.append( (limb_name, "Hold " + str(hold_number), str(frameCount / fps)) )
                break
