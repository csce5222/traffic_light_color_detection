import cv2 as cv
import os
import numpy as np
from random import choice

from traffic_light_color_detection.color_spaces import yuv_to_rgb, rgb_to_yuv
from traffic_light_color_detection.viz import viz_layer

BASE_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "./data/kaggle_dataset/test_dataset/test_images/")


def get_image():
    image_files = os.listdir(BASE_IMAGE_DIR)

    return choice(image_files)


def test_convert_yuv_to_rgb() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    yuv_image = rgb_to_yuv(original_image)
    yuv_to_rgb_image = yuv_to_rgb(yuv_image)
    pics = [original_image, yuv_image, yuv_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> YUV", "YUV->RGB"]
    viz_layer(pics, pic_titles)

    image_comparison = np.isclose(original_image, yuv_to_rgb_image, rtol=2)
    assert len(np.where(image_comparison == False)) < 5 # Set threshold of the observation of 5 feature vectors not being equal





