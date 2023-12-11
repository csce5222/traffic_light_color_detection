from abc import ABC, abstractmethod

import cv2
import copy
import math
import numpy as np
import logging

from traffic_light_color_detection.model import ColorIndex, ColorSpace
from traffic_light_color_detection.color_util import (is_color_equal,
                                                      max_pixel,
                                                      max_pixel_index,
                                                      min_pixel,
                                                      min_pixel_index,
                                                      scale_image,
                                                      un_scale_image)


class ColorSpaceModel(ABC):
    @staticmethod
    @abstractmethod
    def to_color_space(image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def from_color_space(image: np.ndarray) -> np.ndarray:
        pass


class YUVColorSpaceModel(ColorSpaceModel):
    @staticmethod
    def to_color_space(image: np.ndarray) -> np.ndarray:
        mapping = np.array([[0.299, 0.587, 0.114],
                            [-0.147, -0.288, 0.436],
                            [0.615, -0.514, -0.100]])

        yuv = np.dot(image, mapping.T)
        yuv[:, :, 1:] += 128

        return yuv.astype(int)

    @staticmethod
    def from_color_space(image: np.ndarray) -> np.ndarray:
        mapping = np.array([[1.000, 0.000, 1.139],
                            [1.000, -0.394, -0.580],
                            [1.000, 2.032, 0.000]])

        image[:, :, 1:] -= 128
        rgb = np.dot(image, mapping.T)

        return rgb.astype(int)


class YCBCrColorSpaceModel(ColorSpaceModel):
    @staticmethod
    def to_color_space(image: np.ndarray) -> np.ndarray:
        mapping = np.array([[65.481, 128.553, 24.966],
                            [-37.797, -74.203, 112.0],
                            [112.0, -93.786, -18.214]])

        yuv = np.dot(image, mapping.T)
        yuv[:, :, 1:] += 128

        return yuv.astype(int)

    @staticmethod
    def from_color_space(image: np.ndarray) -> np.ndarray:
        mapping = np.array([[0.00456, 0.00, 0.00625],
                            [0.00456, -0.00153, -0.00318],
                            [0.00456, 0.00791, 0.00]])

        image[:, :, 1:] -= 128
        rgb = np.dot(image, mapping.T)

        return rgb.astype(int)


class CMYColorSpaceModal(ColorSpaceModel):
    @staticmethod
    def to_color_space(image: np.ndarray) -> np.ndarray:
        cmy_image = 255 - image
        return cmy_image

    @staticmethod
    def from_color_space(image: np.ndarray) -> np.ndarray:
        rgb_image = 255 - image
        return rgb_image


class XYZColorSpaceModal(ColorSpaceModel):
    @staticmethod
    def get_xyz(red_channel, green_channel, blue_channel):
        trans_matrix = np.array(
            [[0.412453, 0.212671, 0.019334], [0.357580, 0.715160, 0.119193], [0.180423, 0.072169, 0.950227]])

        x = trans_matrix[0][0] * red_channel + trans_matrix[0][1] * green_channel + trans_matrix[0][2] * blue_channel
        y = trans_matrix[1][0] * red_channel + trans_matrix[1][1] * green_channel + trans_matrix[1][2] * blue_channel
        z = trans_matrix[2][0] * red_channel + trans_matrix[2][1] * green_channel + trans_matrix[2][2] * blue_channel

        return (x, y, z)
    @staticmethod
    def to_color_space(image: np.ndarray) -> np.ndarray:
        red_channel = image[:, :, 2]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 0]

        x, y, z = XYZColorSpaceModal.get_xyz(red_channel, green_channel, blue_channel)

        x_normalized = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
        y_normalized = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
        z_normalized = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX)

        merged_xyz = np.dstack((x_normalized, y_normalized, z_normalized)).astype(np.uint8)

        return (merged_xyz)

    @staticmethod
    def from_color_space(image: np.ndarray) -> np.ndarray:
        # Splitting the XYZ image into individual channels
        x, y, z = np.split(image, 3, axis=2)

        trans_matrix = np.array([[3.240479, -1.53715, -0.498535],
                                 [-0.969256, 1.875991, 0.041556],
                                 [0.055648, -0.204043, 1.057311]])

        # Reverse transformation
        r = trans_matrix[0][0] * x + trans_matrix[0][1] * y + trans_matrix[0][2] * z
        g = trans_matrix[1][0] * x + trans_matrix[1][1] * y + trans_matrix[1][2] * z
        b = trans_matrix[2][0] * x + trans_matrix[2][1] * y + trans_matrix[2][2] * z

        # Clip to [0, 255]
        r = np.clip(r, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)

        return cv2.merge((b, g, r))


class LabColorSpaceModal(ColorSpaceModel):
    def inv_lab_sup_func(array):
        modified_arr = np.where(array > 0.206897, array ** 3, (array - 16 / 116) / 7.787)
        return modified_arr

    def lab_sup_func(array):
        modified_arr = np.where(array > 0.008856, array ** 3, (7.787 * array) + (16 / 166))
        return (modified_arr)

    @staticmethod
    def to_color_space(image: np.ndarray) -> np.ndarray:
        delta = 128 if image.dtype in ['uint8'] else 0
        red_channel = image[:, :, 2]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 0]

        x, y, z = XYZColorSpaceModal.get_xyz(red_channel, green_channel, blue_channel)

        Xn = 0.950456  # constant
        Zn = 1.088754  # constant

        x = x / Xn
        z = z / Zn

        L = np.where(y > 0.008856, 116 * (y ** 3 - 16), 903.3 * y)
        a = 500 * (LabColorSpaceModal.lab_sup_func(x) - LabColorSpaceModal.lab_sup_func(y) + delta)
        b = 200 * (LabColorSpaceModal.lab_sup_func(y) - LabColorSpaceModal.lab_sup_func(z) + delta)

        # if image.dtype == 'uint8':
        L = L * (255 / 100)
        a += 128
        b += 128
        # merged_lab = np.dstack((L, a, b)).astype(np.uint8)
        # else:
        merged_lab = np.dstack((L, a, b))

        return (merged_lab)

    @staticmethod
    def from_color_space(image: np.ndarray) -> np.ndarray:
        delta = 128 if image.dtype in ['uint8'] else 0

        L = image[:, :, 0]
        a = image[:, :, 1]
        b = image[:, :, 2]

        y = (L + 16) / 116
        x = a / 500 + y
        z = y - b / 200

        x = LabColorSpaceModal.inv_lab_sup_func(x)
        y = LabColorSpaceModal.inv_lab_sup_func(y)
        z = LabColorSpaceModal.inv_lab_sup_func(z)

        Xn, Zn = 0.950456, 1.088754  # constants

        x = x * Xn
        z = z * Zn

        red_channel = 3.240479 * x - 1.537150 * y - 0.498535 * z
        green_channel = -0.969256 * x + 1.875991 * y + 0.041556 * z
        blue_channel = 0.055648 * x - 0.204043 * y + 1.057311 * z

        # Clip values to the valid range [0, 1]
        red_channel = np.clip(red_channel, 0, 1)
        green_channel = np.clip(green_channel, 0, 1)
        blue_channel = np.clip(blue_channel, 0, 1)

        # Image scaling
        red_channel = (red_channel * 255).astype(np.uint8)
        green_channel = (green_channel * 255).astype(np.uint8)
        blue_channel = (blue_channel * 255).astype(np.uint8)

        # Merging channels as in BGR format
        bgr_image = np.dstack((blue_channel, green_channel, red_channel))

        return bgr_image


class HSVColorSpaceModal(ColorSpaceModel):
    def rgb_to_hsv_on_pixels(pixel: np.ndarray) -> np.ndarray:
        def r(pixel: np.ndarray) -> float:
            return pixel[ColorIndex.RED.value[0]]

        def g(pixel: np.ndarray) -> float:
            return pixel[ColorIndex.GREEN.value[0]]

        def b(pixel: np.ndarray) -> float:
            return pixel[ColorIndex.BLUE.value[0]]

        def hg(pixel: np.ndarray, v: float, g: float) -> float:
            return (v - g) / (v - min(min_pixel(pixel)))

        def hr(pixel: np.ndarray, v: float, r: float) -> float:
            return (v - r) / (v - min(min_pixel(pixel)))

        def hb(pixel: np.ndarray, v: float, b: float) -> float:
            return (v - b) / (v - min(min_pixel(pixel)))

        def ry(pixel: np.ndarray, v: float, g: float) -> float:
            if is_color_equal(ColorIndex.RED.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.BLUE.value, min_pixel_index(pixel)):
                return (1 - hg(pixel, v, g)) / 6
            else:
                return -1.0

        def yg(pixel: np.ndarray, v: float, r: float) -> float:
            if is_color_equal(ColorIndex.GREEN.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.BLUE.value, min_pixel_index(pixel)):
                return (1 + hr(pixel, v, r)) / 6
            else:
                return -1.0

        def gc(pixel: np.ndarray, v: float, b: float) -> float:
            if is_color_equal(ColorIndex.GREEN.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.RED.value, min_pixel_index(pixel)):
                return (3 - hb(pixel, v, b)) / 6
            else:
                return -1.0

        def cb(pixel: np.ndarray, v: float, g: float) -> float:
            if is_color_equal(ColorIndex.BLUE.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.RED.value, min_pixel_index(pixel)):
                return (3 + hg(pixel, v, g)) / 6
            else:
                return -1.0

        def bm(pixel: np.ndarray, v: float, r: float) -> float:
            if is_color_equal(ColorIndex.BLUE.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.GREEN.value, min_pixel_index(pixel)):
                return (5 - hr(pixel, v, r)) / 6
            else:
                return -1.0

        def mr(pixel: np.ndarray, v: float, b: float) -> float:
            if is_color_equal(ColorIndex.RED.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.GREEN.value, min_pixel_index(pixel)):
                return (5 - hb(pixel, v, b)) / 6
            else:
                return -1.0

        def h(pixel: np.ndarray, v: float) -> np.ndarray:
            r_channel = r(pixel)
            g_channel = g(pixel)
            b_channel = b(pixel)

            if (r_channel == g_channel) and (g_channel == b_channel):
                return 1  # h is not defined.  Return abitrary value
            else:
                return max(ry(pixel, v, g_channel),
                           yg(pixel, v, r_channel),
                           gc(pixel, v, b_channel),
                           cb(pixel, v, g_channel),
                           bm(pixel, v, r_channel),
                           mr(pixel, v, b_channel))

        def s(pixel: np.ndarray, v: float) -> float:
            return (v - min(min_pixel(pixel))) / v

        def v(pixel: np.ndarray) -> float:
            return max(max_pixel(pixel))

        logging.debug("HSV: Calculating v channel...")
        v = v(pixel)

        logging.debug("HSV: Calculating h channel...")
        h = h(pixel, v)

        logging.debug("HSV: Calculating s channel...")
        s = s(pixel, v)

        return np.asarray([h, s, v])

    def hsv_to_rgb_on_pixels(pixel: np.ndarray) -> np.ndarray:
        def h(pixel: np.ndarray) -> float:
            return pixel[ColorIndex.HUE.value[0]]

        def s(pixel: np.ndarray) -> float:
            return pixel[ColorIndex.SATURATION.value[0]]

        def v(pixel: np.ndarray) -> float:
            return pixel[ColorIndex.VALUE.value[0]]

        def m(pixel: np.ndarray) -> float:
            return v(pixel) * (1 - s(pixel))

        def F(pixel: np.ndarray) -> float:
            return 6 * h(pixel) - math.floor(6 * h(pixel))

        def n(pixel: np.ndarray) -> float:
            return v(pixel) * (1 - s(pixel) * F(pixel))

        def k(pixel: np.ndarray) -> float:
            return v(pixel) * (1 - s(pixel) * (1 - F(pixel)))

        def ry(pixel: np.ndarray) -> np.ndarray:
            if is_color_equal(ColorIndex.HUE.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.SATURATION.value, min_pixel_index(pixel)):
                return [v(pixel), k(pixel), m(pixel)]
            else:
                return [-1.0, -1.0, -1.0]

        def yg(pixel: np.ndarray) -> float:
            if is_color_equal(ColorIndex.SATURATION.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.VALUE.value, min_pixel_index(pixel)):
                return [n(pixel), v(pixel), m(pixel)]
            else:
                return [-1.0, -1.0, -1.0]

        def gc(pixel: np.ndarray) -> float:
            if is_color_equal(ColorIndex.SATURATION.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.HUE.value, min_pixel_index(pixel)):
                return [m(pixel), v(pixel), k(pixel)]
            else:
                return [-1.0, -1.0, -1.0]

        def cb(pixel: np.ndarray) -> float:
            if is_color_equal(ColorIndex.VALUE.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.HUE.value, min_pixel_index(pixel)):
                return [m(pixel), n(pixel), v(pixel)]
            else:
                return [-1.0, -1.0, -1.0]

        def bm(pixel: np.ndarray) -> float:
            if is_color_equal(ColorIndex.VALUE.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.SATURATION.value, min_pixel_index(pixel)):
                return [k(pixel), m(pixel), v(pixel)]
            else:
                return [-1.0, -1.0, -1.0]

        def mr(pixel: np.ndarray) -> float:
            if is_color_equal(ColorIndex.HUE.value, max_pixel_index(pixel)) and \
                    is_color_equal(ColorIndex.SATURATION.value, min_pixel_index(pixel)):
                return [v(pixel), m(pixel), n(pixel)]
            else:
                return [-1.0, -1.0, -1.0]

        r_g_b = max(ry(pixel), yg(pixel), gc(pixel), cb(pixel), bm(pixel), mr(pixel))

        return np.asarray(r_g_b)

    @staticmethod
    def to_color_space(image: np.ndarray) -> np.ndarray:
        origial_image = copy.deepcopy(image)
        scaled_image = scale_image(origial_image)
        scaled_image = np.apply_along_axis(HSVColorSpaceModal.rgb_to_hsv_on_pixels, axis=-1, arr=scaled_image)
        un_scaled_image = un_scale_image(scaled_image, origial_image)

        return un_scaled_image
    @staticmethod
    def from_color_space(image: np.ndarray) -> np.ndarray:
        origial_image = copy.deepcopy(image)
        scaled_image = scale_image(origial_image)
        scaled_image = np.apply_along_axis(HSVColorSpaceModal.hsv_to_rgb_on_pixels, axis=-1, arr=scaled_image)
        un_scaled_image = un_scale_image(scaled_image, origial_image)

        return un_scaled_image


class HLSColorSpaceModal(ColorSpaceModel):
    @staticmethod
    def to_color_space(image: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def from_color_space(image: np.ndarray) -> np.ndarray:
        H = image[:, :, 0]
        L = image[:, :, 1]
        S = image[:, :, 2]

        C = (1 - np.abs(2 * L - 1)) * S
        X = C * (1 - np.abs((H * 6) % 2 - 1))

        m = L - 0.5 * C

        R1, G1, B1 = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)

        # Conditions based on the hue value (H)
        R1[(0 <= H) & (H < 1 / 6)] = C[(0 <= H) & (H < 1 / 6)]
        G1[(0 <= H) & (H < 1 / 6)] = X[(0 <= H) & (H < 1 / 6)]
        B1[(0 <= H) & (H < 1 / 6)] = 0

        # Combine the adjusted RGB values and scale to 8-bit range
        R = (R1 + m) * 255
        G = (G1 + m) * 255
        B = (B1 + m) * 255

        # Merging channels in BGR format
        bgr_image = np.dstack((B, G, R)).astype(np.uint8)

        return bgr_image


class GrayColorSpaceModel(ColorSpaceModel):
    @staticmethod
    def to_color_space(image: np.ndarray) -> np.ndarray:
        gray_channel = 0.21 * image[:, :, 2] + 0.71 * image[:, :, 1] + 0.08 * image[:, :, 0]
        return gray_channel.astype(np.uint8)

    @staticmethod
    def from_color_space(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        color_image = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                color_image[i, j] = [image[i, j, 0], image[i, j, 1], image[i, j, 2]]

        return color_image


class ColorModelFactory(ColorSpaceModel):
    COLOR_SPACE_MAP = {ColorSpace.CMY.value: CMYColorSpaceModal(),
                       ColorSpace.GRAY.value: GrayColorSpaceModel(),
                       ColorSpace.HLS.value: HSVColorSpaceModal(),
                       ColorSpace.HSV.value: HSVColorSpaceModal(),
                       ColorSpace.Lab.value: LabColorSpaceModal(),
                       ColorSpace.XYZ.value: XYZColorSpaceModal(),
                       ColorSpace.YUV.value: YUVColorSpaceModel(),
                       ColorSpace.YCBCr.value: YCBCrColorSpaceModel()
                       }

    @staticmethod
    def to_color_space(color_space: ColorSpace, image: np.ndarray) -> np.ndarray:
        return ColorModelFactory.COLOR_SPACE_MAP[color_space.value].to_color_space(image)

    @staticmethod
    def from_color_space(color_space: ColorSpace, image: np.ndarray) -> np.ndarray:
        return ColorModelFactory.COLOR_SPACE_MAP[color_space.value].from_color_space(image)

