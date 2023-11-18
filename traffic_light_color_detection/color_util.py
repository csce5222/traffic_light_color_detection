import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List


def max_pixel(pixel: np.ndarray) -> np.ndarray:
    red, green, blue = pixel[0], pixel[1], pixel[2]
    max_pixel = np.max(pixel)

    if red < max_pixel:
        red = 0

    if green < max_pixel:
        green = 0

    if blue < max_pixel:
        blue = 0

    return np.asarray([red, green, blue])


def min_pixel(pixel: np.ndarray) -> np.ndarray:
    red, green, blue = pixel[0], pixel[1], pixel[2]
    min_pixel = np.min(pixel)

    if red > min_pixel:
        red = 0

    if green > min_pixel:
        green = 0

    if blue > min_pixel:
        blue = 0

    return np.asarray([red, green, blue])


def flatten_list(pixel_indices: List) -> List:
    return [channel for pixel_index in pixel_indices for channel in pixel_index]


def max_pixel_index(pixel: np.ndarray) -> List[int]:
    return flatten_list(list(np.where(pixel == np.max(pixel))))


def min_pixel_index(pixel: np.ndarray) -> List[int]:
    return flatten_list(list(np.where(pixel == np.min(pixel))))


def is_color_equal(expected_colors: List, actual_colors: List):
    return all(index in actual_colors for index in expected_colors)


def scale_image(image: np.ndarray) -> np.ndarray:

    scaler = MinMaxScaler()
    color_channels = image.reshape(-1, 3)
    scaled_color_channels = scaler.fit_transform(color_channels)
    return scaled_color_channels.reshape(image.shape)


def un_scale_image(scaled_image: np.ndarray, unscaled_image: np.ndarray) -> np.ndarray:
    return (scaled_image * (unscaled_image.max() - unscaled_image.min() + unscaled_image.min())).astype(np.uint32)

