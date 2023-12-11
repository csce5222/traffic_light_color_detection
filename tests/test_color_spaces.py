import cv2 as cv
import os
from random import choice

from traffic_light_color_detection.model import ColorSpace
from traffic_light_color_detection.color_spaces import ColorModelFactory
from traffic_light_color_detection.viz import viz_layer

BASE_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "./data/kaggle_dataset/test_dataset/test_images/")


def get_image():
    image_files = os.listdir(BASE_IMAGE_DIR)

    return choice(image_files)


def test_convert_rgb_to_yuv() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    yuv_image = ColorModelFactory.to_color_space(ColorSpace.YUV, original_image)
    yuv_to_rgb_image = ColorModelFactory.from_color_space(ColorSpace.YUV, yuv_image)
    rgb_to_yuv_cv_image = cv.cvtColor(original_image, cv.COLOR_BGR2YUV)
    pics = [original_image, yuv_image, rgb_to_yuv_cv_image, yuv_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> YUV", "(CV2) RGB -> YUV", "YUV->RGB"]
    viz_layer(pics, pic_titles)


def test_convert_rgb_to_ycbcr() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    ycbcr_image = ColorModelFactory.to_color_space(ColorSpace.YCBCr, original_image)
    ycbcr_to_rgb_image = ColorModelFactory.from_color_space(ColorSpace.YCBCr, original_image)
    ycbcr_to_rgb_cv_image = cv.cvtColor(original_image, cv.COLOR_BGR2YCrCb)
    pics = [original_image, ycbcr_image, ycbcr_to_rgb_cv_image, ycbcr_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> YCbCr", "(CV2) RGB -> YCrCb", "YCbCr->RGB"]
    viz_layer(pics, pic_titles)


def test_convert_rgb_to_cmy() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    cmy_image = ColorModelFactory.to_color_space(ColorSpace.CMY, original_image)
    cmy_to_rgb_image = ColorModelFactory.from_color_space(ColorSpace.CMY, original_image)

    # TODO - How does CV2 convert to CMY color space???
    pics = [original_image, cmy_image, cmy_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> CMY", "YCbCr->RGB"]
    viz_layer(pics, pic_titles)


def test_convert_rgb_to_xyz() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    xyz_image = ColorModelFactory.to_color_space(ColorSpace.XYZ, original_image)
    xyz_to_rgb_image = ColorModelFactory.from_color_space(ColorSpace.XYZ, original_image)
    xyz_to_rgb_cv_image = cv.cvtColor(original_image, cv.COLOR_RGB2XYZ)
    pics = [original_image, xyz_image, xyz_to_rgb_cv_image, xyz_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> XYZ", "(CV2) RGB -> XYZ", "XYZ->RGB"]
    viz_layer(pics, pic_titles)


def test_convert_rgb_to_lab() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    lab_image = ColorModelFactory.to_color_space(ColorSpace.Lab, original_image)
    lab_to_rgb_image = ColorModelFactory.from_color_space(ColorSpace.Lab, original_image)
    lab_to_rgb_cv_image = cv.cvtColor(original_image, cv.COLOR_RGB2LAB)
    pics = [original_image, lab_image, lab_to_rgb_cv_image, lab_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> Lab", "(CV2) RGB -> XYZ", "Lab->RGB"]
    viz_layer(pics, pic_titles)


def test_convert_rgb_to_hsv() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))
    original_image = cv.resize(original_image, dsize=(200, 200), interpolation=cv.INTER_CUBIC)

    hsv_image = ColorModelFactory.to_color_space(ColorSpace.HSV, original_image)
    hsv_to_rgb_image = ColorModelFactory.from_color_space(ColorSpace.HSV, original_image)
    hsv_to_rgb_cv_image = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
    pics = [original_image, hsv_image, hsv_to_rgb_cv_image, hsv_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> HSV", "(CV2) RGB -> HSV", "RGB -> HSV"]
    viz_layer(pics, pic_titles)


def test_convert_rgb_to_hls() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))
    original_image = cv.resize(original_image, dsize=(200, 200), interpolation=cv.INTER_CUBIC)

    hls_image = ColorModelFactory.to_color_space(ColorSpace.HLS, original_image)
    hsv_to_rgb_image =ColorModelFactory.from_color_space(ColorSpace.HLS, original_image)
    hsv_to_rgb_cv_image = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
    pics = [original_image, hls_image, hsv_to_rgb_cv_image, hsv_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> HLS", "(CV2) RGB -> HLS", "RGB -> HLS"]
    viz_layer(pics, pic_titles)


def test_convert_rgb_to_gray() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    gray_image = ColorModelFactory.to_color_space(ColorSpace.GRAY, original_image)
    gray_to_rgb_image = ColorModelFactory.from_color_space(ColorSpace.GRAY, original_image)
    gray_to_rgb_cv_image =  cv.cvtColor(original_image, cv.COLOR_RGB2GRAY)
    pics = [original_image, gray_image, gray_to_rgb_cv_image, gray_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> GRAY", "(CV2) RGB -> GRAY", "GRAY->RGB"]
    viz_layer(pics, pic_titles)
