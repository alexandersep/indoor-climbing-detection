import cv2 
import matplotlib.pyplot as plt
import numpy as np

import src.image as image
import src.figure as figure
import src.algorithm as algorithm

def adjust_canny(blurred_image, val=0):
    """
    Callback function for trackbars to adjust Canny thresholds.
    Updates the edge-detected image in real-time based on the slider values.
    """
    # Get the current positions of the trackbars
    threshold1 = cv2.getTrackbarPos('Threshold1', 'Canny Edge Detection')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'Canny Edge Detection')

    # Apply Canny edge detection with the current threshold values
    edges = cv2.Canny(blurred_image, threshold1, threshold2)

    #disc = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, disc)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, disc)
    
    # Display the edges in the window
    cv2.imshow('Canny Edge Detection', edges)

def climbImage():
    """
    Algorithm:
        1. Canny Edge Detector
        2. 
    The main application running the project
    Args:
    - void
    Returns:
    - void
    """
    green_hold = image.read("resources/images/backprojection2-green-climb-1.jpg");
    green_climb = image.read("resources/images/green-climb-1.jpg")
    green_climb_gray = cv2.cvtColor(green_climb, cv2.COLOR_BGR2GRAY)
    _ = plt.subplot(1, 2, 1), plt.imshow(green_hold), plt.title("Green Hold")
    _ = plt.subplot(1, 2, 2), plt.imshow(green_climb_gray, cmap='gray'), plt.title("Green Climb")
    figure.wait_close()
    plt.show()

    global blurred_image
    blurred_image = cv2.medianBlur(green_climb_gray, 31)

    cv2.namedWindow('Canny Edge Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Canny Edge Detection', 600, 400)  # Set the window size to be smaller (e.g., 600x400)

    cv2.createTrackbar('Threshold1', 'Canny Edge Detection', 0, 500, lambda val: adjust_canny(blurred_image, val))
    cv2.createTrackbar('Threshold2', 'Canny Edge Detection', 20, 500, lambda val: adjust_canny(blurred_image, val))

    adjust_canny(blurred_image, 0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #figure.display_side_by_side(green_climb, blurred_image, "Green Climb", "Blurred Image")
    #figure.display_side_by_side(green_climb, edges, "Green Climb", "Edge Image")
    
    bins = 12 
    backproj_img, mask, result = algorithm.histogram_and_backprojection(green_hold, green_climb, bins)
    figure.display_four_images(green_hold, mask, backproj_img, result, "Green Hold", "Mask", "Backprojection", "Detected Areas")

    green_climb_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.medianBlur(green_climb_gray, 31)

    cv2.namedWindow('Canny Edge Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Canny Edge Detection', 600, 400)  # Set the window size to be smaller (e.g., 600x400)

    cv2.createTrackbar('Threshold1', 'Canny Edge Detection', 0, 500, lambda val: adjust_canny(blurred_image, val))
    cv2.createTrackbar('Threshold2', 'Canny Edge Detection', 100, 500, lambda val: adjust_canny(blurred_image, val))

    adjust_canny(blurred_image, 0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.namedWindow('Filled Contours', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Filled Contours', 600, 400)  # Set the window size to be smaller (e.g., 600x400)

    # Find contours from the Canny edges
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank image to fill the contours
    filled_image = np.zeros_like(green_climb)  # Same size as input image, black background
    
    # Fill each contour (using cv2.drawContours to fill them)
    cv2.drawContours(filled_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Show the result
    cv2.imshow('Filled Contours', filled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def app():
    """
    The main application running the project
    Args:
    - void
    Returns:
    - void
    """
    #figure.setup()
    #climbImage()
    #inrange()
    #task_sift_matching()
    #edge_detect_numbers()
    video()

def video():
    cap = cv2.VideoCapture("resources/videos/green-climb.mp4")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('resources/videos/output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue
        img_copy = img.copy()
        #red_image = inrange(img)
        #cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        #cv2.imshow('Original Image', red_image)
        #cv2.resizeWindow('Original Image', 600, 400)  # Set the window size to be smaller (e.g., 600x400)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(img_gray, (5, 5), 1)

        circles = cv2.HoughCircles(blurred_image, 
                                    cv2.HOUGH_GRADIENT, dp=1, minDist=20, 
                                    param1=150, param2=30, minRadius=10, maxRadius=25)
        
        circle_list = []
        if circles is not None:
            # Convert the circle coordinates to integers
            unpacked_circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in unpacked_circles:
                # Draw the outer circle
                #cv2.circle(img, (x, y), r, (0, 255, 0), 4)
                found = (x,y)
                circle_list.append(found)

        # Load the image
        green_climb = img_copy
    
        # Convert the image from BGR to HSV
        hsv_image = cv2.cvtColor(green_climb, cv2.COLOR_BGR2HSV)
        
        # Define the range for detecting green color in HSV
        lower_green = np.array([35, 50, 50])  # Lower bound for green
        upper_green = np.array([85, 255, 255])  # Upper bound for green

        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        disc = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, disc)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, disc)

        # Display the results
        #cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        #cv2.imshow('Original Image', green_mask)
        #cv2.resizeWindow('Original Image', 600, 400)  # Set the window size to be smaller (e.g., 600x400)

        contours, _ = cv2.findContours(image=green_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        if not contours:
            print("No contours found")
            return

        holds = []  # List to store closest circle-contour pairs

        # Loop over each contour
        for (circle_x, circle_y) in circle_list:

            min_dist = float("inf")  # Initialize with a large number
            closest_contour = None  # To store the closest circle coordinates
            is_contour = False

            # Loop over each circle to find the minimum distance
            for contour in contours:
                contour_x, contour_y = calculate_centroid(contour)
                temp_dist = dist(circle_x, circle_y, contour_x, contour_y)
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    closest_contour = contour
                    is_contour = True

            if is_contour:
                # Store the pair of contour and closest circle
                holds.append(((circle_x, circle_y), closest_contour))

        # Draw the lines between the contour centroids and the closest circles
        #for circle_coords, contour in holds:
            #circle_x, circle_y = circle_coords
            #contour_x, contour_y = calculate_centroid(contour)
            #print(f"Circle ({circle_x}, {circle_y}), Contour ({contour_x}, {contour_y})")
            #cv2.line(img, (circle_x, circle_y), (contour_x, contour_y), (0, 0, 255), 5)

        for contour in contours:
            x, y = calculate_centroid(contour)
            cv2.circle(img, (x, y), 5, (0, 255, 0), 3)

        contour_end = calculate_hold(holds, isEnd=True, k=0)
        x, y, w, h = cv2.boundingRect(contour_end)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(img, "End Hold", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        contour_begin = calculate_hold(holds, isEnd=False, k=0)
        x, y, w, h = cv2.boundingRect(contour_begin)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(img, "Start Hold", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the result
        #show_image(img)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 600, 400)  # Set the window size to be smaller (e.g., 600x400)
        cv2.imshow('frame', img)
        out.write(img)
        c = cv2.waitKey(1)
        if c & 0xff == ord('q'):
            break

    cap.release()
    out.release()

def calculate_hold(holds, isEnd, k):
    """
    kth hold
    """
    highest = find_highest_hold(holds)

    min_dist = float("inf")
    closest_hold = None
    
    close_holds = []
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
                    close_holds.append(closest_hold)
        else:
            if contour_y < circle_y:
                temp_dist = dist(circle_x, circle_y, contour_x, contour_y)
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    closest_hold = contour
                    close_holds.append(closest_hold)
    return close_holds[len(close_holds) - k - 1]

def find_highest_hold(holds):
    lowest_x = float('inf')
    lowest_y = float('inf') 
    for _, contour in holds:
        x, y = calculate_centroid(contour)
        if lowest_y > y:
            lowest_x = x
            lowest_y = y
    return (lowest_x, lowest_y)

def dist(x1, y1, x2, y2):
    return np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

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

def edge_detection(image, thresh1, thresh2):
    """
    Image in grayscale, compute Canny edge
    """
    blurred_image = cv2.medianBlur(image, 5)
    edges = cv2.Canny(blurred_image, thresh1, thresh2)

    disc = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, disc)
    return edges

def show_image(image):
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    # Show the result
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_spr(contour):
    spr = algorithm.SPR(contour)
    print("Area =", spr.area())
    print("Length =", spr.length())
    print("Width =", spr.width())
    print("elongatedness =", spr.elongatedness())
    print("solidity = ", spr.solidity())
    print("perimeter = ", spr.perimeter())
    #print("convex_hull =", spr.convex_hull())
    #print("convex min rect ratio =", spr.minimum_bounding_rectangle())

def edge_detect_numbers():
    number = cv2.imread("resources/images/number-sign-1-2-cropped-green-climb-1.jpg", cv2.IMREAD_GRAYSCALE)
    edges = edge_detection(number, 150, 100)

    contours, _ = cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("No contours found")
        return

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(number)
    cv2.drawContours(image=mask, contours=[largest], contourIdx=-1, color=255, thickness=cv2.FILLED, lineType=cv2.LINE_AA)

    calculate_spr(largest)

    mask_floodfill = mask.copy()
    cv2.floodFill(mask_floodfill, None, (0, 0), 255)  # Flood fill from corner

    mask_filled = cv2.bitwise_or(mask, cv2.bitwise_not(mask_floodfill))

    cv2.namedWindow('Numbers', cv2.WINDOW_NORMAL)
    cv2.imshow('Numbers', mask_filled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("resources/largest_filled_mask.jpg", mask_filled)

def task_sift_matching():
    number = cv2.imread("resources/images/number-sign-1-cropped-green-climb-1.jpg", cv2.IMREAD_GRAYSCALE)
    climb = cv2.imread("resources/images/green-climb-1.jpg", cv2.IMREAD_GRAYSCALE)

    good_matches, keypoints_left, keypoints_right, matched_image = detect_and_match_features(number, climb)

    # Source: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    # Calculating xl_coords, and xr_coords
    xl_coords = []
    xr_coords = []
    for match in good_matches:
        xl_coords.append(keypoints_left[match.queryIdx].pt[0]) # queryIdx for left_image's good matches
        xr_coords.append(keypoints_right[match.trainIdx].pt[0]) # trainIdx for right_images's good matches
    
    print("Sample x-coordinates in left image (xL):", xl_coords[:10])
    print("Sample x-coordinates in right image (xR):", xr_coords[:10])
    
    # Display the matched image
    plt.figure(figsize=(15, 10))
    plt.imshow(matched_image)
    plt.title("Feature Matches Between Left and Right Images")
    plt.show()
 
def detect_and_match_features(left_image, right_image, ratio_test_threshold=0.7):
    """
    Detects and matches SIFT features between two images using FLANN-based matcher and Lowe's ratio test.

    Parameters:
    - left_image: Left image in stereo pair (grayscale).
    - right_image: Right image in stereo pair (grayscale).
    - ratio_test_threshold: Threshold for Lowe's ratio test to filter matches (default=0.7).

    Returns:
    - good_matches: List of good matches that passed the ratio test.
    - keypoints_left: Keypoints in the left image.
    - keypoints_right: Keypoints in the right image.
    - matched_image: Image showing matches between left and right images.
    """

    # Write your code will be here
    # Source: https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    # Source: https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html

    # Detect the keypoints using SIFT Detector, compute the descriptors
    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)
    keypoints_left, descriptors_left = sift.detectAndCompute(left_image, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(right_image, None)

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SIFT is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_left, descriptors_right, 2)

    #-- Filter matches using the Lowe's ratio test
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_test_threshold * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key = lambda x:x.distance)
    #-- Draw matches
    matched_image = np.empty((max(left_image.shape[0], right_image.shape[0]), left_image.shape[1]+right_image.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(left_image, keypoints_left, right_image, keypoints_right, good_matches, matched_image, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return good_matches,keypoints_left, keypoints_right, matched_image

def inrange(green_climb):
    # Load the image
    #green_climb = cv2.imread("resources/images/green-climb-1.jpg")
    
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(green_climb, cv2.COLOR_BGR2HSV)
    
    # Define the range for detecting green color in HSV
    #lower_green = np.array([35, 50, 50])  # Lower bound for green
    #upper_green = np.array([85, 255, 255])  # Upper bound for green

    #lower_red1 = np.array([0, 120, 70])   # Lower bound for the first red range (near 0 hue)
    #upper_red1 = np.array([10, 255, 255]) # Upper bound for the first red range
    
    #lower_red2 = np.array([170, 120, 70])  # Lower bound for the second red range (near 180 hue)
    #upper_red2 = np.array([180, 255, 255]) # Upper bound for the second red range

    # Define color ranges in HSV
    red_lower = np.array([170, 50, 50])  # Adjust the values based on #9a3241, #78343a
    red_upper = np.array([180, 255, 255])
        
    white_lower = np.array([0, 0, 150])  # Adjust the values based on #cd9194
    white_upper = np.array([180, 50, 255])
    
    # Create a binary mask where green pixels are 255 and others are 0
    #green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    mask1 = cv2.inRange(hsv_image, red_lower, red_upper)
    mask2 = cv2.inRange(hsv_image, white_lower, white_upper)

    green_mask = cv2.bitwise_or(mask1, mask2)
    
    # Optionally, you can use the mask to extract the green regions
    green_region = cv2.bitwise_and(green_climb, green_climb, mask=green_mask)

    return green_region
    
    # Display the results
    #cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    #cv2.imshow('Original Image', green_climb)
    #cv2.resizeWindow('Original Image', 600, 400)  # Set the window size to be smaller (e.g., 600x400)

    #cv2.namedWindow('Green Mask', cv2.WINDOW_NORMAL)
    #cv2.imshow('Green Mask', green_mask)
    #cv2.resizeWindow('Green Mask', 600, 400)  # Set the window size to be smaller (e.g., 600x400)

    #cv2.namedWindow('Detected Green Region', cv2.WINDOW_NORMAL)
    #cv2.imshow('Detected Green Region', green_region)
    #cv2.resizeWindow('Detected Green Region', 600, 400)  # Set the window size to be smaller (e.g., 600x400)
    #
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
