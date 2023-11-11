import numpy as np
import cv2

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


def get_xyz(red_channel, green_channel, blue_channel):
  trans_matrix = np.array([[0.412453, 0.212671, 0.019334], [0.357580, 0.715160, 0.119193], [0.180423, 0.072169, 0.950227]])

  x = trans_matrix[0][0]*red_channel + trans_matrix[0][1]*green_channel + trans_matrix[0][2]*blue_channel
  y = trans_matrix[1][0]*red_channel + trans_matrix[1][1]*green_channel + trans_matrix[1][2]*blue_channel
  z = trans_matrix[2][0]*red_channel + trans_matrix[2][1]*green_channel + trans_matrix[2][2]*blue_channel

  return(x, y, z)

def bgr_to_xyz(image):

  red_channel = image[:, :, 2]
  green_channel = image[:, :, 1]
  blue_channel = image[:, :, 0]

  x, y, z = get_xyz(red_channel, green_channel, blue_channel)

  x_normalized = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
  y_normalized = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
  z_normalized = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX)

  merged_xyz = np.dstack((x_normalized, y_normalized, z_normalized)).astype(np.uint8)

  return (merged_xyz)