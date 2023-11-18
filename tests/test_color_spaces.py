import cv2 as cv
import os
import numpy as np
from random import choice

from traffic_light_color_detection.color_spaces import cmy_to_rgb, \
                                                       gray_to_rgb, \
                                                       rgb_to_cmy, \
                                                       rgb_to_gray, \
                                                       rgb_to_yuv, \
                                                       rgb_to_ycbcr, \
                                                       ycbcr_to_rgb, \
                                                       yuv_to_rgb
from traffic_light_color_detection.viz import viz_layer

BASE_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "./data/kaggle_dataset/test_dataset/test_images/")


def get_image():
    image_files = os.listdir(BASE_IMAGE_DIR)

    return choice(image_files)


def test_convert_rgb_to_yuv() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    rgb_to_yuv_image = rgb_to_ycbcr(original_image)
    yuv_to_rgb_image = yuv_to_rgb(rgb_to_yuv_image)
    # rgb_to_yuv_cv_image = cv.cvtColor(original_image, cv.COLOR2RGBYU)
    # pics = [original_image, rgb_to_yuv_image, rgb_to_yuv_cv_image, rgb_to_yuv_cv_image]
    # pic_titles = ["Original Image", "RGB-> YUV", "(CV2) RGB -> YUV", "YUV->RGB"]
    # viz_layer(pics, pic_titles)

    image_comparison = np.isclose(original_image, yuv_to_rgb_image, rtol=2)
    assert len(np.where(image_comparison == False)) < 5 # Set threshold of the observation of 5 feature vectors not being equal


def test_convert_rgb_to_ycbcr() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    ycbcr_image = rgb_to_ycbcr(original_image)
    ycbcr_to_rgb_image = ycbcr_to_rgb(ycbcr_image)
    # ycbcr_to_rgb_cv_image = cv.cvtColor(original_image, cv.COLOR_BGR2YCrCb)
    # pics = [original_image, ycbcr_image, ycbcr_to_rgb_cv_image, ycbcr_to_rgb_image]
    # pic_titles = ["Original Image", "RGB-> YCbCr", "(CV2) RGB -> YCrCb", "YCbCr->RGB"]
    # viz_layer(pics, pic_titles)

    image_comparison = np.isclose(original_image, ycbcr_to_rgb_image, rtol=2)
    assert len(np.where(image_comparison == False)) < 5 # Set threshold of the observation of 5 feature vectors not being equal


def test_convert_rgb_to_cmy() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    cmy_image = rgb_to_cmy(original_image)
    cmy_to_rgb_image = cmy_to_rgb(cmy_image)

    # TODO - How does CV2 convert to CMY color space???
    pics = [original_image, cmy_image, cmy_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> CMY", "YCbCr->RGB"]
    viz_layer(pics, pic_titles)

    image_comparison = np.isclose(original_image, cmy_to_rgb_image, rtol=2)
    assert len(np.where(image_comparison == False)) < 5 # Set threshold of the observation of 5 feature vectors not being equal


def test_convert_rgb_to_grey() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    cmy_image = rgb_to_gray(original_image)
    gray_to_rgb_image = gray_to_rgb(cmy_image, original_image)
    gray_to_rgb_cv_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    pics = [original_image, cmy_image, gray_to_rgb_cv_image, gray_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> GRAY", "(CV2) RGB -> CMY", "GRAY->RGB"]
    viz_layer(pics, pic_titles)

    image_comparison = np.isclose(original_image, gray_to_rgb_image, rtol=2)
    assert len(np.where(image_comparison == False)) < 5 # Set threshold of the observation of 5 feature vectors not being equal

