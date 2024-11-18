# image.py
# Helper function for working with images

import cv2

def read(img_path):
    """
    Read an image as a string of path
    Args:
    - img_path: str of path name to image 

    Returns:
    - img: cv2 image in rgb format
    """
    img = cv2.imread(img_path);
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise ValueError("Failed to read image at path {} ", img_path)

    return img

def first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    _, img = cap.read()
    cap.release()
    return img

