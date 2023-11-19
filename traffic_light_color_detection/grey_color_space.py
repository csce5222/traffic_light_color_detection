import numpy as np


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    gray_channel = 0.21 * image[:, :, 2] + 0.71 * image[:, :, 1] + 0.08 * image[:, :, 0]
    return gray_channel.astype(np.uint8)


def gray_to_rgb(grayscale_image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    height, width = grayscale_image.shape[:2]
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            color_image[i, j] = [original_image[i, j, 0], original_image[i, j, 1], original_image[i, j, 2]]

    return color_image
