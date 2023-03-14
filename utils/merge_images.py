import cv2
import argparse

import numpy


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
    return cv2.addWeighted(background, alpha, face, beta, gamma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Merge two images together.",
    )
    parser.add_argument("background_path", help="File path to the background image")
    parser.add_argument("face_path", help="File path to the face image")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Weight of the background image (default: 0.4)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Weight of the face image (default: 0.1)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0, help="Brightness adjustment (default: 0)"
    )
    args = parser.parse_args()

    # Merge the images using the specified arguments
    merged_image = merge_images(
        args.background_path, args.face_path, args.alpha, args.beta, args.gamma
    )

    # Display the merged image
    cv2.imshow("Merged", merged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
