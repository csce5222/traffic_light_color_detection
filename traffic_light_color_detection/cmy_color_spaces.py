import numpy as np


def rgb_to_cmy(image: np.ndarray) -> np.ndarray:
    cmy_image = 255 - image
    return cmy_image


def cmy_to_rgb(image: np.ndarray) -> np.ndarray:
    rgb_image = 255 - image
    return rgb_image