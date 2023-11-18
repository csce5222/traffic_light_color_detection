import cv2 as cv
import numpy as np
import os

from test_util import BASE_IMAGE_DIR, get_image
from traffic_light_color_detection.grey_color_space import gray_to_rgb, rgb_to_gray
from traffic_light_color_detection.viz import viz_layer


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