import numpy as np

def bgr_to_hls(image):
  red_channel = image[:, :, 2]
  green_channel = image[:, :, 1]
  blue_channel = image[:, :, 0]

  Vmax = np.maximum(red_channel, np.maximum(green_channel, blue_channel))
  Vmin = np.minimum(red_channel, np.minimum(green_channel, blue_channel))

  L = (Vmax + Vmin) / 2
  S = np.where(L < 0.5, (Vmax - Vmin) / (Vmax + Vmin), (Vmax - Vmin) / (2 - (Vmax + Vmin)))

  delta = Vmax - Vmin

  H = np.zeros_like(Vmax)

  H[Vmax == red_channel] = ((green_channel - blue_channel) / delta)[Vmax == red_channel]
  H[Vmax == green_channel] = (2.0 + (blue_channel - red_channel) / delta)[Vmax == green_channel]
  H[Vmax == blue_channel] = (4.0 + (red_channel - green_channel) / delta)[Vmax == blue_channel]
  H[delta == 0] = 0.0
  H[H < 0] += 360

  if image.dtype == 'uint8':
    H = H / 2
    S = S * 255
    L = L * 255
  elif image.dtype == 'int16':
    H = -H
    S = -S
    V = -L

  H = (H + 6.0) % 6.0 / 6.0

  return np.dstack((H, S, L))

