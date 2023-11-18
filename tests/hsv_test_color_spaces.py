import cv2 as cv
import os

from traffic_light_color_detection.color_spaces import hsv_to_rgb, \
                                                       rgb_to_hsv
from test_util import BASE_IMAGE_DIR, get_image


def test_convert_rgb_to_hsv() -> None:
    # image_file_path = get_image()
    image_file_path = "00796.jpg"
    original_image = cv.imread(os.path.join(BASE_IMAGE_DIR, image_file_path))
    original_image = cv.resize(original_image, dsize=(200, 200), interpolation=cv.INTER_CUBIC)
    rgb_to_hsv_image = rgb_to_hsv(original_image)
    hsv_to_rgb_cv_image = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
    hsv_to_rgb_image = hsv_to_rgb(rgb_to_hsv_image)
    # pics = [original_image, hsv_to_rgb_cv_image]
    # pics = [original_image, rgb_to_hsv_image, hsv_to_rgb_cv_image, hsv_to_rgb_image]
    # pic_titles = ["Original Image", "(CV2) RGB -> HSV"]
    # pic_titles = ["Original Image", "RGB-> HSV", "(CV2) RGB -> HSV", "RGB -> HSV"]
    # viz_layer(pics, pic_titles)
