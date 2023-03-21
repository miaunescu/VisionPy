import argparse

import cv2
import numpy

from utils.common import display_image


def merge_images(
    background_path: str, face_path: str, alpha: float, beta: float, gamma: float
) -> numpy.ndarray:
    """
    Merges two images together by adding weighted pixel values from each image.

    :param:
        background_path (str): The file path to the background image.
        face_path (str): The file path to the face image.
        alpha (float): The weight of the first image. A larger value of alpha means that
            the first image will have a greater impact on the final blended image.
        beta (float): The weight of the second image. A larger value of beta means that
            the second image will have a greater impact on the final blended image.
        gamma (float): A scalar added to the weighted sum of pixel values. It can be used
            to adjust the brightness of the final image.

     :return:
        merged_image (numpy.ndarray): The merged image as a numpy array.

    Raises:
        cv2.error: If there is an error reading the input images.
    """
    try:
        background = cv2.imread(background_path)
        face = cv2.imread(face_path)
    except cv2.error as e:
        raise e(f"Error: Could not read input images \n {e.message}")

    # Resize images to the same dimensions
    if background.shape != face.shape:
        face = cv2.resize(face, (background.shape[1], background.shape[0]))

    # Merge the images with the given weighting parameters
    return cv2.addWeighted(face, alpha, face, beta, gamma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Merge two images together.",
    )
    parser.add_argument(
        "-bg",
        "--background_path",
        default="images/harry_theme/uniforms/harry.jpg",
        help="File path to the background image",
    )
    parser.add_argument(
        "-fp",
        "--face_path",
        default="images/mario_theme/roi.png",
        help="File path to the face image",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.4,
        help="Weight of the background image (default: 0.4)",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=0.1,
        help="Weight of the face image (default: 0.1)",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0,
        help="Brightness adjustment (default: 0)",
    )
    args = parser.parse_args()

    merged_image = merge_images(
        args.background_path, args.face_path, args.alpha, args.beta, args.gamma
    )

    display_image(title="Merged Image", image=merged_image)
