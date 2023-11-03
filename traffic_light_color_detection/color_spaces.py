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

