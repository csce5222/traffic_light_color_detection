from enum import Enum
import numpy as np
from typing import List


class TrainPixel(object):

    def __init__(self, image: np.ndarray, max_color: int, indices: List):
        self.image = image
        self.color = max_color
        self.indices = indices


class ColorMinxMax(Enum):
    MIN_COLOR = "minimum"
    MAX_COLOR = "maxmimum"
