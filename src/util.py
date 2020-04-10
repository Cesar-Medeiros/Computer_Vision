
import cv2 as cv
from enum import Enum

def relative_error(num1, num2):
    return abs(num1-num2)/max(num1, num2)

def center_cnt(cnt):
    M = cv.moments(cnt)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])

    return cX, cY

def getApproxContour(cnt):
    # between 1% and 5%
    epsilon = 0.04*cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    return approx


class Shape(Enum):
    CIRCLE = 0
    TRIANGLE = 3
    RECTANGLE = 4
    OCTOGON = 8
    INVALID = -1

    @classmethod
    def _missing_(cls, value):
        return Shape.INVALID

def getShapeName(shape):
    return {
        Shape.CIRCLE: "Circle",
        Shape.TRIANGLE: "Triangle",
        Shape.RECTANGLE: "Rectangle",
        Shape.OCTOGON: "Octogon",
    }.get(shape, str(shape))



class Color(Enum):
    RED = 1
    BLUE = 2

def getColorName(color):
    return {
        Color.RED: "Red",
        Color.BLUE: "Blue",
    }.get(color, str(color))



def checkValid(color, shape):
    return  (color == Color.RED and (shape == Shape.TRIANGLE  or shape == Shape.OCTOGON or shape == Shape.CIRCLE)) \
        or (color == Color.BLUE and (shape == Shape.CIRCLE or shape == Shape.RECTANGLE))

def getMax(array):

    maximum = -1
    index = -1

    for i, elem in enumerate(array):
        if (elem > maximum):
            maximum = elem
            index = i

    return maximum, index