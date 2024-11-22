import cv2 
import numpy as np

from src.utils import dist

def process_hands(frame, limb_list, contours):
    min_contours = []
    for limb, limb_name in limb_list:
        if limb_name == 'LEFT_INDEX' or limb_name == 'RIGHT_INDEX':
            limb_x, limb_y = limb # Usually left index finger
            min_dist = float('inf')
            for contour in contours:
                x, y = calculate_centroid(contour)
                distance = dist(x, y, limb_x, limb_y)
                if min_dist > distance and distance < 100: # Ensure the hand is close to the hold
                    min_dist = distance
                    min_contours.append( (contour, limb_name) )

    for (min_contour, limb_name) in min_contours: 
        color = (0, 0, 0)
        if limb_name == 'LEFT_INDEX':
            color = (255, 0, 0)
        if limb_name == 'RIGHT_INDEX':
            color = (0, 0, 255)
        x, y, w, h = cv2.boundingRect(min_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
    return frame

def process_holds(frame):
    circle_list = get_circles(frame)
    # Green mask processing
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    disc = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, disc)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, disc)

    contours, _ = cv2.findContours(image=green_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    holds = []
    for (circle_x, circle_y) in circle_list:
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

    for contour in contours:
        x, y = calculate_centroid(contour)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), 3)

    contour_end = calculate_hold(holds, isEnd=True)
    x, y, w, h = cv2.boundingRect(contour_end)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv2.putText(frame, "End Hold", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    contour_begin = calculate_hold(holds, isEnd=False)
    x, y, w, h = cv2.boundingRect(contour_begin)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv2.putText(frame, "Start Hold", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return contours

def get_circles(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(frame_gray, (5, 5), 1)
    
    circles = cv2.HoughCircles(blurred_image, 
                                cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                                param1=150, param2=30, minRadius=10, maxRadius=25)
    
    circle_list = []
    if circles is not None:
        unpacked_circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in unpacked_circles:
            found = (x, y)
            circle_list.append(found)
    return circle_list


def calculate_hold(holds, isEnd):
    highest = find_highest_hold(holds)

    min_dist = float("inf")
    closest_hold = None
    
    # Loop through each hold to find the closest one with a positive slope
    for circle_coords, contour in holds:
        circle_x, circle_y = circle_coords
        contour_x, contour_y = calculate_centroid(contour)
        
        # Ensure that the contour has a positive slope towards the circle
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

# Function to calculate and draw the centroid of a contour
def calculate_centroid(contour):
    # Step 1: Calculate the moments of the contour
    moments = cv2.moments(contour)
    
    # Step 2: Calculate the x and y coordinates of the centroid
    if moments['m00'] != 0:  # To avoid division by zero
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = 0, 0
    
    return cx, cy
