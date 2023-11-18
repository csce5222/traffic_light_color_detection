import copy
import math
import numpy as np
import logging

from traffic_light_color_detection.model import ColorIndex
from traffic_light_color_detection.color_util import (is_color_equal,
                                                      max_pixel,
                                                      max_pixel_index,
                                                      min_pixel,
                                                      min_pixel_index,
                                                      scale_image,
                                                      un_scale_image)


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


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    origial_image = copy.deepcopy(image)
    scaled_image = scale_image(origial_image)
    scaled_image = np.apply_along_axis(rgb_to_hsv_on_pixels, axis=-1, arr=scaled_image)
    un_scaled_image = un_scale_image(scaled_image, origial_image)

    return un_scaled_image


def hsv_to_rgb_on_pixels(pixel: np.ndarray) -> np.ndarray:
    def h(pixel: np.ndarray) -> float:
        return pixel[ColorIndex.HUE.value[0]]

    def s(pixel: np.ndarray) -> float:
        return pixel[ColorIndex.SATURATION.value[0]]

    def v(pixel: np.ndarray) -> float:
        return pixel[ColorIndex.VALUE.value[0]]

    def m(pixel: np.ndarray) -> float:
        return v(pixel) * (1 - s(pixel))

    def F(pixel:np.ndarray) -> float:
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


def hsv_to_rgb(image: np.ndarray) -> np.ndarray:
    origial_image = copy.deepcopy(image)
    scaled_image = scale_image(origial_image)
    scaled_image = np.apply_along_axis(hsv_to_rgb_on_pixels, axis=-1, arr=scaled_image)
    un_scaled_image = un_scale_image(scaled_image, origial_image)

    return un_scaled_image
