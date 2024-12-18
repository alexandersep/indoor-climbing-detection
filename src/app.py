import cv2 

from src.hold_processing import *
from src.body_processing import *
from src.opticalflow_processing import *
from src.utils import *

video_path = "resources/videos/green-climb.mp4"
output_path = "resources/videos/output/green-climb-trimmed.mp4"

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
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        contours, start_hold, end_hold = process_holds(frame)
        limb_list = process_skeleton(frame, mp_pose, pose, mp_drawing)
        frame, isStarted, isFinished = process_hands(frame, limb_list, contours, start_hold, end_hold)

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
        frameCount += 1

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

    download_video_in_range(video_path, output_path, beginFrame, endFrame)

    video.release()
    cv2.destroyAllWindows()
