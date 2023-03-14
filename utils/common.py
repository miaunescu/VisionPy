import cv2
import numpy


def display_image(title: str, image: numpy.ndarray):
    """
    :param title:The title of the window in which the image will be displayed.
    :param image: The image data to be displayed.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
