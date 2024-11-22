import cv2
import numpy as np

# Source: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

def process_opticalflow(current_frame, limb_list, lk_params, isPersonInFrame, mask, old_gray, color, p0=[]):
    if not isPersonInFrame or len(limb_list) == 0:
        if len(limb_list) > 0:
            p0 = np.array(limb_list, dtype=np.float32).reshape(-1, 1, 2)
            isPersonInFrame = True
        return p0, old_gray, mask, isPersonInFrame, current_frame

    frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            current_frame = cv2.circle(current_frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        current_frame = cv2.add(current_frame, mask)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    else:
        isPersonInFrame = False

    return p0, old_gray, mask, isPersonInFrame, current_frame

def setup_optical_flow(video):
    """
    Video must be valid
    """
    frame = None
    if video.isOpened():
        _, frame = video.read()
    # params for ShiTomasi corner detection
    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    last_frame = frame

    mask = np.zeros_like(last_frame)
    old_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    p0 = []

    isPersonInFrame = False
    return lk_params, mask, old_gray, color, p0, isPersonInFrame

def dense_optical_flow(last_frame, current_frame):
    """
    Pixel by pixel basis, this is extremely expensive and produces like 1 FPS
    """
    last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(last_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(last_frame)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    dense_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return dense_flow
