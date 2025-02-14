import cv2 
import numpy as np

from src.utils import dist

def process_hands(frame, limb_list, contours, start_hold, end_hold):
    HITBOX_PADDING = 25
    min_contours = []
    for limb, limb_name in limb_list:
        if limb_name == 'LEFT_INDEX' or limb_name == 'RIGHT_INDEX' or limb_name == 'LEFT_FOOT_INDEX' or limb_name == 'RIGHT_FOOT_INDEX':
            limb_x, limb_y = limb
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if x - HITBOX_PADDING <= limb_x <= x + w + HITBOX_PADDING and y - HITBOX_PADDING <= limb_y <= y + h + HITBOX_PADDING:
                    min_contours.append( (contour, limb_name) )

    left_hand_hold = ((-1, -1), (-1, -1))
    right_hand_hold = ((-1, -1), (-1, -1))
    left_foot_hold = ((-1, -1), (-1, -1))
    right_foot_hold = ((-1, -1), (-1, -1))
    started_left, started_right = False, False
    finished_left, finished_right = False, False
    for (min_contour, limb_name) in min_contours: 
        # x, y = calculate_centroid(min_contour)
        x, y, w, h = cv2.boundingRect(min_contour)
        if limb_name == 'LEFT_INDEX':
            color = (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            left_hand_hold = ((x, y), (w, h))
        if limb_name == 'RIGHT_INDEX':
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            right_hand_hold = ((x, y), (w, h))
        if limb_name == 'LEFT_FOOT_INDEX':
            color = (125, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            left_foot_hold = ((x, y), (w, h))
        if limb_name == 'RIGHT_FOOT_INDEX':
            color = (0, 0, 125)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            right_foot_hold = ((x, y), (w, h))

        (sx, sy) = start_hold
        if (sx - 50 <= x <= sx + 50) and (sy - 50 <= y <= sy + 50):
            if limb_name == 'LEFT_INDEX':
                started_left = True
            if limb_name == 'RIGHT_INDEX':
                started_right = True

        (ex, ey) = end_hold
        if (ex - 50 <= x <= ex + 50) and (ey - 50 <= y <= ey + 50):
            if limb_name == 'LEFT_INDEX':
                finished_left = True
            if limb_name == 'RIGHT_INDEX':
                finished_right = True

    return frame, (started_left, started_right), (finished_left, finished_right), left_hand_hold, right_hand_hold, left_foot_hold, right_foot_hold

def process_holds(frame):
    circle_list = get_circles(frame)
    # Green mask processing
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    disc = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, disc)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, disc)

    contours, _ = cv2.findContours(image=green_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    holds = []
    for (circle_x, circle_y, _) in circle_list:
        min_dist = float("inf")
        closest_contour = None
        is_contour = False

        for contour in contours:
            contour_x, contour_y = calculate_centroid(contour)
            temp_dist = dist(circle_x, circle_y, contour_x, contour_y)
            if temp_dist < min_dist:
                min_dist = temp_dist
                closest_contour = contour
                is_contour = True

        if is_contour:
            holds.append(((circle_x, circle_y), closest_contour))

    contour_begin = calculate_hold(holds, isEnd=False)
    x, y = cv2.boundingRect(contour_begin)[:2]
    start_hold = (x,y)

    contour_end = calculate_hold(holds, isEnd=True)
    x, y = cv2.boundingRect(contour_end)[:2]

    end_hold = (x, y)

    return contours, start_hold, end_hold

def get_circles(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(frame_gray, (5, 5), 1)
    
    circles = cv2.HoughCircles(blurred_image, 
                                cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                                param1=150, param2=30, minRadius=10, maxRadius=25)
    if circles is not None:
        return np.round(circles[0, :]).astype("int")
    else:
        return None


def calculate_hold(holds, isEnd):
    highest = find_highest_hold(holds)

    min_dist = float("inf")
    closest_hold = None
    
    for circle_coords, contour in holds:
        circle_x, circle_y = circle_coords
        contour_x, contour_y = calculate_centroid(contour)
        if isEnd:
            if contour_y < circle_y and highest == (contour_x, contour_y):
                temp_dist = dist(circle_x, circle_y, contour_x, contour_y)
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    closest_hold = contour
        else:
            if contour_y < circle_y:
                temp_dist = dist(circle_x, circle_y, contour_x, contour_y)
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    closest_hold = contour
    return closest_hold

def find_highest_hold(holds):
    lowest_x = float('inf')
    lowest_y = float('inf') 
    for _, contour in holds:
        x, y = calculate_centroid(contour)
        if lowest_y > y:
            lowest_x = x
            lowest_y = y
    return (lowest_x, lowest_y)

def calculate_centroid(contour):
    moments = cv2.moments(contour)
    
    if moments['m00'] != 0:  # To avoid division by zero
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = 0, 0
    
    return cx, cy
