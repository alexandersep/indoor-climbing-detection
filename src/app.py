import cv2 

from src.hold_processing import *
from src.body_processing import *
from src.opticalflow_processing import *
from src.utils import *

video_path = "resources/videos/green-climb.mp4"
output_path = "resources/videos/output/green-climb-trimmed.mp4"
output_path_select_frame = "resources/videos/output/green-climb-trimmed-select.mp4"

def app():
    video, frame_width, frame_height, _, _ = load_video(video_path)
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

    left_hand_holds = []
    right_hand_holds = []
    left_foot_holds = []
    right_foot_holds = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        contours, start_hold, end_hold = process_holds(frame)
        limb_list = process_skeleton(frame, mp_pose, pose, mp_drawing)
        frame, isStarted, isFinished, left_hand_hold, right_hand_hold, left_foot_hold, right_foot_hold = process_hands(frame, limb_list, contours, start_hold, end_hold)
        frameCount += 1

        if started and not finished:
            skip_holds(left_hand_holds, left_hand_hold, frameCount)
            skip_holds(right_hand_holds, right_hand_hold, frameCount)
            skip_holds(left_foot_holds, left_foot_hold, frameCount)
            skip_holds(right_foot_holds, right_foot_hold, frameCount)

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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if beginFrame >= 10:
        beginFrame -= 10
    if endFrame >= 10:
        endFrame -= 10

    print("Climber has started climb? {}".format(started))
    print("Climber has completed climb? {}".format(finished))
    print("Climber starts at frame {}".format(beginFrame))
    print("Climber ends at frame {}".format(endFrame))

    print("Left Hand holds {}".format(left_hand_holds))
    print("Right Hand holds {}".format(right_hand_holds))

    print("Left Foot holds {}".format(left_foot_holds))
    print("Right Foot holds {}".format(right_foot_holds))

    print("Number of left hand holds {}".format(len(left_hand_holds)))
    print("Number of right hand holds {}".format(len(right_hand_holds)))
    print("Number of right foot holds {}".format(len(left_foot_holds)))
    print("Number of right foot holds {}".format(len(right_foot_holds)))

    video.release()
    cv2.destroyAllWindows()

    download_video_in_range(video_path, output_path, beginFrame, endFrame, left_hand_holds, right_hand_holds, left_foot_holds, right_foot_holds)
    (_, _, selectFrameNumber) = left_hand_holds[7] # selecting the nth left hand hold. Note: I do not bounds check
    download_video_in_range(video_path, output_path_select_frame, selectFrameNumber, endFrame, left_hand_holds, right_hand_holds, left_foot_holds, right_foot_holds)

def skip_holds(holds, hold, frameCount):
    ((x, y), (w, h)) = hold
    skip_hand = False
    if hold == ((-1, -1), (-1, -1)):
        return
    for left_hand in holds:
        ((limb_x, limb_y), _, _) = left_hand
        distance = dist(x, y, limb_x, limb_y)
        if distance <= 100:
            skip_hand = True

    if not skip_hand:
        holds.append( ((x, y), (w, h), frameCount) )
