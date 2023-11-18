from color_spaces import get_xyz
import numpy as np

def bgr_to_lab(image):
  delta = 128 if image.dtype in ['uint8'] else 0
  red_channel = image[:, :, 2]
  green_channel = image[:, :, 1]
  blue_channel = image[:, :, 0]

  x, y, z = get_xyz(red_channel, green_channel, blue_channel)

  Xn = 0.950456   # constant
  Zn = 1.088754   # constant

  x = x / Xn
  z = z / Zn

  L = np.where(y > 0.008856, 116 * (y**3 - 16), 903.3 * y)
  a = 500 * (lab_sup_func(x) - lab_sup_func(y) + delta)
  b = 200 * (lab_sup_func(y) - lab_sup_func(z) + delta)

  # if image.dtype == 'uint8':
  L = L * (255 / 100)
  a += 128
  b += 128
  # merged_lab = np.dstack((L, a, b)).astype(np.uint8)
  # else:
  merged_lab = np.dstack((L, a, b))

  return(merged_lab)

def lab_sup_func(array):
  modified_arr = np.where(array > 0.008856, array**3, (7.787 * array) + (16 / 166))
  return (modified_arr)