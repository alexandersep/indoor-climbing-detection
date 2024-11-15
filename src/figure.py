# figure.py
# Helper function for working with figures/plt

import matplotlib.pyplot as plt

def close(event):
    """
    Interrupt like function which closes a plt
    based on str value of event
    Args:
    - event: str with keyboard name
    Returns:
    - void
    """
    if event.key == "escape":
        plt.close(event.canvas.figure)

def wait_close():
    """
    Wrapper to close a plot based on function figure_close
    Args:
    - event: str with keyboard name
    Returns:
    - void
    """
    plt.gcf().canvas.mpl_connect('key_press_event', close)

def setup():
    """
    Setup plt with figure height and width and other setup related functions
    Args:
    - void
    Returns:
    - void
    """
    plt.figure(figsize=(10, 8))
    plt.tight_layout(pad=5.0, w_pad=5.0, h_pad=5.0)

def display_side_by_side(img0, img1, img0_text="First Image", img1_text="Second Image"):
    """
    Displays two images side by side
    Args:
    - img0: 2D numpy array of image
    - img1: 2D numpy array of image
    - img0_text: str text corresponding to img0
    - img1_text: str text corresponding to img1
    Returns:
    - void
    """
    _ = plt.subplot(1, 2, 1), plt.imshow(img0), plt.title(img0_text)
    _ = plt.subplot(1, 2, 2), plt.imshow(img1), plt.title(img1_text)
    wait_close()
    plt.show()

def display_four_images(img0, img1, img2, img3, img0_text="First Image", img1_text="Second Image", img2_text="Third Image", img3_text="Fourth Image"):
    """
    Displays four images side by side
    Args:
    - img0: 2D numpy array of image
    - img1: 2D numpy array of image
    - img0_text: str text corresponding to img0
    - img1_text: str text corresponding to img1
    Returns:
    - void
    """
    _ = plt.subplot(2, 2, 1), plt.imshow(img0), plt.title(img0_text)
    _ = plt.subplot(2, 2, 2), plt.imshow(img1), plt.title(img1_text)
    _ = plt.subplot(2, 2, 3), plt.imshow(img2), plt.title(img2_text)
    _ = plt.subplot(2, 2, 4), plt.imshow(img3), plt.title(img3_text)
    wait_close()
    plt.show()
