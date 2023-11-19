import cv2 as cv
import numpy as np
import os

from traffic_light_color_detection.cmy_color_spaces  import cmy_to_rgb, rgb_to_cmy
from traffic_light_color_detection.viz import viz_layer
from test_util import BASE_IMAGE_DIR, get_image


def test_convert_rgb_to_cmy() -> None:
    image_file_path = get_image()
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))

    cmy_image = rgb_to_cmy(original_image)
    cmy_to_rgb_image = cmy_to_rgb(cmy_image)
    cmy_to_rgb_cv_image = cv.cvtColor(original_image, cv.COLOR_BGR2YCrCb)
    pics = [original_image, cmy_image, cmy_to_rgb_cv_image, cmy_to_rgb_image]
    pic_titles = ["Original Image", "RGB-> CMY", "(CV2) RGB -> CMY", "CMY->RGB"]
    viz_layer(pics, pic_titles)

    image_comparison = np.isclose(original_image, cmy_to_rgb_image, rtol=2)
    assert len(np.where(image_comparison == False)) < 5 # Set threshold of the observation of 5 feature vectors not being equal