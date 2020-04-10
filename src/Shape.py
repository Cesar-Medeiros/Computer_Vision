import util
import math
import cv2 as cv


class Shape:
    def __init__(self, cnt, color):
        self.cnt = cnt
        self.color = color
        self.shape = None
        self.identifyShape()


    def isCircle(self):
        PI = 3.14
        perimeter = cv.arcLength(self.cnt, True)
        area = cv.contourArea(self.cnt)

        radius_p = perimeter/(2*PI)
        radius_a = math.sqrt(area/PI)

        return (util.relative_error(radius_p, radius_a) < 0.01)


    def identifyShape(self):
        if self.isCircle():
            self.shape = util.Shape.CIRCLE
        else:
            approx = util.getApproxContour(self.cnt)
            self.shape = util.Shape(len(approx))
            self.cnt = approx

        return self.shape, self.cnt


    def isValid(self):
        return util.checkValid(self.color, self.shape)
