import cv2 
import matplotlib.pyplot as plt
import numpy as np

import src.image as image
import src.figure as figure
import src.algorithm as algorithm

from sklearn.cluster import MeanShift

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
    figure.setup()
    #climbImage()
    inrange()

def inrange():
    # Load the image
    green_climb = cv2.imread("resources/images/green-climb-1.jpg")
    
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(green_climb, cv2.COLOR_BGR2HSV)
    
    # Define the range for detecting green color in HSV
    lower_green = np.array([35, 50, 50])  # Lower bound for green
    upper_green = np.array([85, 255, 255])  # Upper bound for green
    
    # Create a binary mask where green pixels are 255 and others are 0
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Optionally, you can use the mask to extract the green regions
    green_region = cv2.bitwise_and(green_climb, green_climb, mask=green_mask)
    
    # Display the results
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image', green_climb)
    cv2.resizeWindow('Original Image', 600, 400)  # Set the window size to be smaller (e.g., 600x400)

    cv2.namedWindow('Green Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Green Mask', green_mask)
    cv2.resizeWindow('Green Mask', 600, 400)  # Set the window size to be smaller (e.g., 600x400)

    cv2.namedWindow('Detected Green Region', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Green Region', green_region)
    cv2.resizeWindow('Detected Green Region', 600, 400)  # Set the window size to be smaller (e.g., 600x400)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
