import cv2 as cv
import numpy as np
import math
import util
import Shape

class ColorRange:
    def __init__(self, lower, upper):
        self.lower = np.array(lower)
        self.upper = np.array(upper)

class HoughCircles:
    def __init__(self, img):
        self.img = img
    
    def getCircles(self):
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        _, _, v = cv.split(hsv)
        circles = getCircleCountours(v)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

        return circles

class ColorDetection:
    def __init__(self, img):
        self.img = img
    
    def getColor(self):
        pass
        

    

class ShapeDetection:

    def __init__(self, color_name, arr_colors):
        self.color = color_name
        self.arr_colors = arr_colors
        
    
    def segment_color(self, img):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        
        for color_range in self.arr_colors:
            lower = color_range.lower
            upper = color_range.upper
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            mask += cv.inRange(hsv, lower, upper)

        mask = preprocessing_mask(mask)
        return mask


    def getShape(self, mask, useHough):

        shape_arr = []

        # find contours
        contours = getContours(mask)
        contours = preprocessing_contours(contours)

        for cnt in contours:
            shape = Shape.Shape(cnt, self.color)
            if shape.isValid():
                shape_arr.append(shape)

        return shape_arr



    def outputResult(self, shape_arr, img):
        for shape in shape_arr:
            cv.drawContours(img, [shape.cnt], -1, (0, 255, 0), 2)
            x, y = util.center_cnt(shape.cnt)
            label = util.getColorName(shape.color) + ' ' + util.getShapeName(shape.shape)
            cv.putText(img, label, (x - 40, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    def outputEdges(self, shape_arr, img):
        for shape in shape_arr:
            cv.drawContours(img, [shape.cnt], -1, (255, 255, 255), 1)



def preprocessing_mask(mask):
    height, width = mask.shape

    # Remove small noise
    mask = cv.medianBlur(mask, 3)

    # Increase border to prevent dilate with canvas edges
    border_size = 50
    mask = cv.copyMakeBorder(mask, border_size, border_size, border_size, border_size, cv.BORDER_CONSTANT)

    # Close mask gaps
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Remove border
    mask = mask[border_size:height + border_size, border_size:width + border_size]

    return mask


def getContours(edges):
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def preprocessing_contours(contours):
    new_contours = []

    for cnt in contours:

        # close open contours
        hull = cv.convexHull(cnt)

        # discard noise
        if cv.contourArea(hull) > 1000:
            new_contours.append(hull)

    return new_contours


def getCircleCountours(gray):
    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT, 1, 120, param1=100, param2=50, minRadius=0, maxRadius=0)
    return circles
    
