import argparse

import cv2

from utils.common import display_image

MAX_PIXEL_VALUE = 255.0


def overlay_images(foreground_path: str, background_path: str, alpha_path: str):
    """
    Overlays a foreground image on a background image using an alpha mask.
     :param:foreground_path (str): The path to the foreground image file.
     :param:background_path (str): The path to the background image file.
     :param: alpha_path (str): The path to the alpha mask image file.
    """
    # Read the images
    foreground = cv2.imread(foreground_path)
    background = cv2.imread(background_path)
    alpha = cv2.imread(alpha_path)

    # Check that images were read successfully
    if any(img is None for img in [foreground, background, alpha]):
        raise ValueError("One or more images could not be read")

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    """
    Dividing the image by 255 is a normalization step commonly used in image processing. 
    In this case, the out_image variable is the result of adding the foreground and background 
    images together after they have been multiplied by their respective alpha masks. 
    The result is a matrix of floating-point numbers that represents the intensity of each pixel in the final image.
    The maximum value of a pixel in an 8-bit color image (such as a JPEG or PNG file) is 255. 
    By dividing the matrix by 255, we scale the intensity values to the range of 0 to 1, 
    which is a common range used in image processing algorithms. 
    This normalization step can improve the performance of subsequent operations on the image, 
    such as filtering, segmentation, or feature extraction.
    """
    alpha = alpha.astype(float) / MAX_PIXEL_VALUE

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    out_image = cv2.add(foreground, background)

    # Display the output image
    """
    Additionally, cv2.imshow expects pixel values to be in the range of 0 to 255. 
    Dividing the image by 255 before displaying it ensures that the pixel values are in the 
    expected range and can be displayed correctly.
    """
    display_image(title="Output Image", image=out_image / MAX_PIXEL_VALUE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Overlay images",
    )
    parser.add_argument(
        "--foreground_path",
        type=str,
        default="../images/harry_theme/glasses/glasses.png",
        help="File path to the foreground image",
    )
    parser.add_argument(
        "--background_path",
        type=str,
        default="harry.png",
        help="File path to the background image",
    )
    parser.add_argument(
        "--alpha_path",
        type=str,
        default="puppets_alpha.png",
        help="File path to the alpha_path image",
    )
    args = parser.parse_args()
    overlay_images(args.foreground_path, args.background_path, args.alpha_path)
