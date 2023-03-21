import cv2
import argparse

from utils.common import display_image


def resize_image(image_path: str, scale_percent: int):
    """
    Load an image from the specified file path and resize it by the specified scale percent
    :param image_path: Path to the image file to resize.
    :param scale_percent: The scale percent to resize the image by.
    """
    # Load the image from the file path
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    except cv2.error as e:
        raise e(f"Failed to load image from {image_path} \n {e.message}")

    # todo: shall we replace print statements with logger functions?
    print(f"Original Dimensions :{img.shape}")

    # Calculate the new dimensions based on the scale percent
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dimension = (width, height)

    # Resize the image
    resized = cv2.resize(src=img, dsize=dimension, interpolation=cv2.INTER_AREA)
    if resized is None:
        raise ValueError(f"Failed to resize image from {image_path} to {dimension}")

    print(f"Resized Dimensions : {resized.shape}")

    # Display the resized image
    display_image(title="Resized image", image=resized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Resize an image",
    )
    parser.add_argument(
        "--image_path",
        default="/home/img/python.png",
        type=str,
        help="Path to the image file",
    )
    parser.add_argument(
        "--scale_percent", type=int, default=60, help="Scale percent as an integer"
    )
    args = parser.parse_args()

    # Resize the image using the specified arguments
    resize_image(args.image_path, args.scale_percent)
