import numpy as np


def rgb_to_yuv(image: np.ndarray) -> np.ndarray:
    mapping = np.array([[0.299, 0.587, 0.114],
                        [-0.147, -0.288, 0.436],
                        [0.615, -0.514, -0.100]])

    yuv = np.dot(image, mapping.T)
    yuv[:, :, 1:] += 128

    return yuv.astype(int)


def yuv_to_rgb(image: np.ndarray) -> np.ndarray:
    mapping = np.array([[1.000, 0.000, 1.139],
                        [1.000, -0.394, -0.580],
                        [1.000, 2.032, 0.000]])

    image[:, :, 1:] -= 128
    rgb = np.dot(image, mapping.T)

    return rgb.astype(int)


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    mapping = np.array([[65.481, 128.553, 24.966],
                        [-37.797, -74.203, 112.0],
                        [112.0, -93.786, -18.214]])

    yuv = np.dot(image, mapping.T)
    yuv[:, :, 1:] += 128

    return yuv.astype(int)


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    mapping = np.array([[0.00456, 0.00, 0.00625],
                        [0.00456, -0.00153, -0.00318],
                        [0.00456, 0.00791, 0.00]])

    image[:, :, 1:] -= 128
    rgb = np.dot(image, mapping.T)

    return rgb.astype(int)


def rgb_to_cmy(image: np.ndarray) -> np.ndarray:
    cmy_image = 255 - image
    return cmy_image


def cmy_to_rgb(image: np.ndarray) -> np.ndarray:
    rgb_image = 255 - image
    return rgb_image