# backproject.py
# Functions for performing backprojection

import cv2
import numpy as np

# Source: https://docs.opencv.org/4.x/da/d7f/tutorial_back_projection.html
# Modified for our use case
def histogram_and_backprojection(img0, img1, bins):
    """
    Creates a histogram of img0 and backprojects it on img1 with bins
    Args:
    - img0: str of path name to the image 
    - img1: str of path name to the image 
    - bins: int number of bins
    Returns:
    - backproj: cv2 image in rgb format
    - mask: 2D numpy array binary image 
    - result: 2D numpy array of mask AND'd with img1
    """
    histSize = max(bins, 2)
    ranges = [0, 180] # hue_range
    
    ch = (0, 0)
    hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    hue = np.empty(hsv.shape, hsv.dtype)
    cv2.mixChannels([hsv], [hue], ch)
 
    hist = cv2.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hue1 = np.empty(hsv1.shape, hsv1.dtype)
    cv2.mixChannels([hsv1], [hue1], ch)
 
    backproj = cv2.calcBackProject([hue1], [0], hist, ranges, scale=1)

    _, mask = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)

     # Optional: Apply some morphological operations to clean up the mask
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disc)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disc)

    # Apply the mask to the test image to highlight detected areas
    result = cv2.bitwise_and(img1, img1, mask=mask)

    backproj = cv2.cvtColor(backproj, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return backproj, mask, result

class SPR:
    def __init__(self, contour):
        self.contour = contour

    def area(self):
        return cv2.contourArea(self.contour)

    def perimeter(self):
        return cv2.arcLength(self.contour, True)

    def length_width_ratio(self):
        length = self.length()
        width = self.width()
        return length / width

    def rectangularity(self):
        return self.area() / (self.length() * self.width())
    
    def minimum_bounding_rectangle(self):
        return cv2.minAreaRect(self.contour)

    def elongatedness(self):
        _, (width, height), _ = self.minimum_bounding_rectangle()
        return max(width, height) / min(width, height)

    def solidity(self):
        hull = self.convex_hull()
        hull_area = cv2.contourArea(hull) 
        return self.area() / hull_area

    def convex_hull(self):
        return cv2.convexHull(self.contour)

    def length(self):
        _, _, width, height = cv2.boundingRect(self.contour)
        return max(width, height)

    def width(self):
        _, _, width, height = cv2.boundingRect(self.contour)
        return min(width, height)
