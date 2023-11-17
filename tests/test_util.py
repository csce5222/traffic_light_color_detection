import numpy as np
import os
from hypothesis import given, strategies as st
from random import choice, randint
from typing import List

from traffic_light_color_detection.color_util import is_color_equal, max_pixel, max_pixel_index, min_pixel, min_pixel_index
from traffic_light_color_detection.model import Color, ColorIndex

from model import ColorMinxMax, TrainPixel

BASE_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "./data/kaggle_dataset/test_dataset/test_images/")


def generate_pixel(red: int, green: int, blue: int) -> np.ndarray:
    return np.asarray([red, green, blue])


def calculate_max_color(max_color: Color) -> TrainPixel:
    if max_color == Color.RED:
        red = randint(171, 255)
        green = randint(0, 85)
        blue = randint(86, 170)
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([red, 0, 0]), ColorIndex.RED.value)
    elif max_color == Color.GREEN:
        red = randint(0, 85)
        green = randint(171, 255)
        blue = randint(86, 170)
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([0, green, 0]), ColorIndex.GREEN.value)
    elif max_color == Color.BLUE:
        red = randint(0, 85)
        green = randint(86, 170)
        blue = randint(171, 255)
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([0, 0, blue]), ColorIndex.BLUE.value)
    elif max_color == Color.RED_GREEN:
        red = randint(171, 255)
        green = red
        blue = randint(86, 170)
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([red, green, 0]), ColorIndex.RED_GREEN.value)
    elif max_color == Color.RED_BLUE:
        red = randint(171, 255)
        green = randint(86, 170)
        blue = red
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([red, 0, blue]), ColorIndex.RED_BLUE.value)
    elif max_color == Color.GREEN_BLUE:
        red = randint(86, 170)
        green = randint(171, 255)
        blue = green
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([0, green, blue]), ColorIndex.GREEN_BLUE.value)
    elif max_color == Color.RED_GREEN_BLUE:
        red = randint(171, 255)
        green = red
        blue = red
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([red, green, blue]), ColorIndex.RED_GREEN_BLUE.value)


def calculate_min_color(min_color: Color) -> TrainPixel:
    if min_color == Color.RED:
        red = randint(0, 85)
        green = randint(171, 255)
        blue = randint(86, 170)
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([red, 0, 0]), ColorIndex.RED.value)
    elif min_color == Color.GREEN:
        red = randint(171, 255)
        green = randint(0, 85)
        blue = randint(86, 170)
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([0, green, 0]), ColorIndex.GREEN.value)
    elif min_color == Color.BLUE:
        red = randint(171, 255)
        green = randint(86, 170)
        blue = randint(0, 85)
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([0, 0, blue]), ColorIndex.BLUE.value)
    elif min_color == Color.RED_GREEN:
        red = randint(86, 170)
        green = red
        blue = randint(171, 255)
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([red, green, 0]), ColorIndex.RED_GREEN.value)
    elif min_color == Color.RED_BLUE:
        red = randint(86, 170)
        green = randint(171, 255)
        blue = red
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([red, 0, blue]), ColorIndex.RED_BLUE.value)
    elif min_color == Color.GREEN_BLUE:
        red = randint(171, 255)
        green = randint(86, 170)
        blue = green
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([0, green, blue]), ColorIndex.GREEN_BLUE.value)
    elif min_color == Color.RED_GREEN_BLUE:
        red = randint(171, 255)
        green = red
        blue = red
        return TrainPixel(generate_pixel(red, green, blue), np.asarray([red, green, blue]), ColorIndex.RED_GREEN_BLUE.value)


@st.composite
def pixel_strategy(draw, max_color=None, min_color=None) -> TrainPixel:
    if max_color:
        return calculate_max_color(max_color)
    elif min_color:
        return calculate_min_color(min_color)


@given(pixel_strategy(max_color=Color.RED))
def test_max_rgb_filter_max_red(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MAX_COLOR)

    assert assert_pixel_indices(train_pixel, max_pixel_index(train_pixel.image))


@given(pixel_strategy(max_color=Color.GREEN))
def test_max_rgb_filter_max_green(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MAX_COLOR)

    assert assert_pixel_indices(train_pixel, max_pixel_index(train_pixel.image))


@given(pixel_strategy(max_color=Color.BLUE))
def test_max_rgb_filter_max_blue(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MAX_COLOR)

    assert assert_pixel_indices(train_pixel, max_pixel_index(train_pixel.image))


@given(pixel_strategy(max_color=Color.RED_BLUE))
def test_max_rgb_filter_max_red_and_blue(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MAX_COLOR)

    assert assert_pixel_indices(train_pixel, max_pixel_index(train_pixel.image))



@given(pixel_strategy(max_color=Color.RED_GREEN))
def test_max_rgb_filter_max_red_and_green(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MAX_COLOR)

    assert assert_pixel_indices(train_pixel, max_pixel_index(train_pixel.image))



@given(pixel_strategy(max_color=Color.GREEN_BLUE))
def test_max_rgb_filter_max_green_and_blue(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MAX_COLOR)

    assert assert_pixel_indices(train_pixel, max_pixel_index(train_pixel.image))


@given(pixel_strategy(max_color=Color.RED_GREEN_BLUE))
def test_max_rgb_filter_max_red_and_green_and_blue(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MAX_COLOR)

    assert assert_pixel_indices(train_pixel, max_pixel_index(train_pixel.image))


@given(pixel_strategy(min_color=Color.RED))
def test_min_rgb_filter_min_red(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MIN_COLOR)

    assert assert_pixel_indices(train_pixel, min_pixel_index(train_pixel.image))


@given(pixel_strategy(min_color=Color.GREEN))
def test_min_rgb_filter_min_green(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MIN_COLOR)

    assert assert_pixel_indices(train_pixel, min_pixel_index(train_pixel.image))



@given(pixel_strategy(min_color=Color.BLUE))
def test_min_rgb_filter_min_blue(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MIN_COLOR)

    assert assert_pixel_indices(train_pixel, min_pixel_index(train_pixel.image))



@given(pixel_strategy(min_color=Color.RED_BLUE))
def test_min_rgb_filter_min_red_and_blue(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MIN_COLOR)

    assert assert_pixel_indices(train_pixel, min_pixel_index(train_pixel.image))



@given(pixel_strategy(min_color=Color.RED_GREEN))
def test_min_rgb_filter_min_red_and_green(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MIN_COLOR)

    assert assert_pixel_indices(train_pixel, min_pixel_index(train_pixel.image))



@given(pixel_strategy(min_color=Color.GREEN_BLUE))
def test_min_rgb_filter_min_green_and_blue(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MIN_COLOR)

    assert assert_pixel_indices(train_pixel, min_pixel_index(train_pixel.image))



@given(pixel_strategy(min_color=Color.RED_GREEN_BLUE))
def test_min_rgb_filter_min_red_and_green_and_blue(train_pixel: TrainPixel) -> None:
    assert_pixel(train_pixel, ColorMinxMax.MIN_COLOR)

    assert assert_pixel_indices(train_pixel, min_pixel_index(train_pixel.image))


def assert_pixel(pixel: TrainPixel, min_max_color: ColorMinxMax):
    if min_max_color == ColorMinxMax.MAX_COLOR:
        assert np.array_equal(max_pixel(pixel.image), pixel.color)
    elif min_max_color == ColorMinxMax.MIN_COLOR:
        assert np.array_equal(min_pixel(pixel.image), pixel.color)


def assert_pixel_indices(pixel: TrainPixel, pixel_indices: List) -> bool:
    return is_color_equal(pixel.indices, pixel_indices)


def get_image():
    image_files = os.listdir(BASE_IMAGE_DIR)

    return choice(image_files)