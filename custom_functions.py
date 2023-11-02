import cv2
import numpy as np

def bgr_to_xyz(image):

  red_channel = image[:, :, 2]
  green_channel = image[:, :, 1]
  blue_channel = image[:, :, 0]

  trans_matrix = np.array([[0.412453, 0.212671, 0.019334], [0.357580, 0.715160, 0.119193], [0.180423, 0.072169, 0.950227]])

  x = trans_matrix[0][0]*red_channel + trans_matrix[0][1]*green_channel + trans_matrix[0][2]*blue_channel
  y = trans_matrix[1][0]*red_channel + trans_matrix[1][1]*green_channel + trans_matrix[1][2]*blue_channel
  z = trans_matrix[2][0]*red_channel + trans_matrix[2][1]*green_channel + trans_matrix[2][2]*blue_channel

  x_normalized = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
  y_normalized = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
  z_normalized = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX)

  merged_xyz = np.dstack((x_normalized, y_normalized, z_normalized)).astype(np.uint8)

  return (merged_xyz)