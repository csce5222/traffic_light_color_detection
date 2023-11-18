from enum import Enum


class Color(Enum):
    RED = "R"
    GREEN = "G"
    BLUE = "B"
    RED_GREEN = "RG"
    RED_BLUE = "RB"
    GREEN_BLUE = "GB"
    RED_GREEN_BLUE = "RGB"


class ColorIndex(Enum):
    RED = [0]
    GREEN = [1]
    BLUE = [2]
    RED_GREEN = [0, 1]
    RED_BLUE = [0, 2]
    GREEN_BLUE = [1, 2]
    RED_GREEN_BLUE = [0, 1, 2]
    HUE = [0]
    SATURATION = [1]
    VALUE = [2]

