import src.image as image
import src.figure as figure
import src.backproject as backproject

def app():
    """
    The main application running the project
    Args:
    - void
    Returns:
    - void
    """
    figure.setup()

    green_climb = image.read("resources/images/green-climb-1.jpg");
    green_hold = image.read("resources/images/backprojection2-green-climb-1.jpg");
    figure.display_side_by_side(green_hold, green_climb, "Green Hold", "Green climb")

    bins = 12 
    backproj_img, mask, result = backproject.histogram_and_backprojection(green_hold, green_climb, bins)
    figure.display_side_by_side(backproj_img, mask, "Backprojection", "Mask")
    figure.display_side_by_side(green_hold, result, "Test Image", "Detected Areas")
