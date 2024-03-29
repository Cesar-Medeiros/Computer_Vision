#!/usr/bin/env python

'''
Signal Recognition

Usage:
   vcom_p1.py
'''

# Python 2/3 compatibility
from __future__ import print_function

import sys
import numpy as np
import math
import argparse
import cv2 as cv

IMAGES_DIR = 'images/'
IMAGES_SIMPLE_DIR = IMAGES_DIR + 'simple/'
IMAGES_COMPLEX_DIR = IMAGES_DIR + 'complex/'
IMAGES_OTHER_DIR = IMAGES_DIR + 'other/'

IMAGE_SCALE_WIDTH = 500
IMAGE_SCALE_HEIGHT = 500


def load_image(fn):
    fn = cv.samples.findFile(fn)
    print('\tLoading "%s" ...' % fn)
    img = cv.imread(fn, cv.IMREAD_UNCHANGED)
    return img


def capture_frame():
    cap = cv.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        cv.imshow('Webcam - Press Q to select frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            return frame


def hsvRedSegmentation(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # first range of red
    lower_red = np.array([0, 200, 75])
    upper_red = np.array([5, 255, 255])
    mask1 = cv.inRange(hsv, lower_red, upper_red)

    # second range of red
    lower_red = np.array([170, 200, 75])
    upper_red = np.array([180, 255, 255])
    mask2 = cv.inRange(hsv, lower_red, upper_red)

    # combining masks
    mask = mask1 + mask2
    return mask


def hsvBlueSegmentation(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_blue = np.array([104, 200, 75])
    upper_blue = np.array([126, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    return mask


def getEdges(gray):
    return cv.Canny(gray, threshold1=50, threshold2=200)


def getContours(edges):
    contours, _ = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def getCircleCountours(gray):
    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT, 1, 120, param1=100, param2=50, minRadius=0, maxRadius=0)
    return circles


def getApproxContour(cnt):
    # between 1% and 5%
    epsilon = 0.04*cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    return approx


def relative_error(num1, num2):
    return abs(num1-num2)/max(num1, num2)


def isCircle(cnt):
    PI = 3.14

    perimeter = cv.arcLength(cnt, True)
    area = cv.contourArea(cnt)

    radius_p = perimeter/(2*PI)
    radius_a = math.sqrt(area/PI)

    return (relative_error(radius_p, radius_a) < 0.01)


def getShapeName(numVertices):
    text = ""
    if numVertices == 3:
        text = "Triangle"
    elif numVertices == 4:
        text = "Rectangle"
    elif numVertices == 8:
        text = "Octogon"
    else:
        text = str(numVertices)

    return text


def identifyShape(cnt):
    shape_name = ""
    shape = None

    if isCircle(cnt):
        shape_name = "Circle"
        shape = cnt
    else:
        approx = getApproxContour(cnt)
        shape_name = getShapeName(len(approx))
        shape = approx

    return shape_name, shape


def center_cnt(cnt):
    M = cv.moments(cnt)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])

    return cX, cY


def value_eq(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    new_v = cv.equalizeHist(v)
    hsv_eq = cv.merge([h, s, new_v])
    bgr_eq = cv.cvtColor(hsv_eq, cv.COLOR_HSV2BGR)
    return bgr_eq


def saturation_eq(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    new_s = cv.equalizeHist(s)
    hsv_eq = cv.merge([h, new_s, v])
    bgr_eq = cv.cvtColor(hsv_eq, cv.COLOR_HSV2BGR)
    return bgr_eq


def saturation_eq2(img, gamma):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    new_s = np.array(255*(s/255)**gamma, dtype='uint8')
    hsv_eq = cv.merge([h, new_s, v])
    bgr_eq = cv.cvtColor(hsv_eq, cv.COLOR_HSV2BGR)
    return bgr_eq


def increase_sat(img, alpha, beta):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    new_s = cv.convertScaleAbs(s, alpha=alpha, beta=beta)
    hsv_eq = cv.merge([h, new_s, v])
    bgr_eq = cv.cvtColor(hsv_eq, cv.COLOR_HSV2BGR)
    return bgr_eq


def stretch_sat(img, min, max):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    new_s = test(s, min, max)

    hsv_eq = cv.merge([h, new_s, v])
    bgr_eq = cv.cvtColor(hsv_eq, cv.COLOR_HSV2BGR)
    return bgr_eq


def show_sat(img, text):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    cv.imshow(text, s)


def test(img1, min_val, max_val):

    # Create zeros array to store the stretched image
    minmax_img = np.zeros((img1.shape[0], img1.shape[1]), dtype='uint8')

    # min_val = np.min(img1)
    # max_val = np.max(img1)
    delta = max_val - min_val

    # Loop over the image and apply Min-Max formulae
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            new_val = 255*(img1[i, j]-min_val)/delta
            new_val = max(0, min(new_val, 255))
            minmax_img[i, j] = new_val

    return minmax_img


def optionMenu():
    choice = input("""
        For circle detection, please select one of the following methods:

        A: Simple Shape Detection using Contour approximation
        B: Circle Hough Transform

        ==> """)

    choice = choice.lower()

    if choice != "a" and choice != "b":
        print("\n\tYou must only select either A or B")
        print("\tPlease try again!\n")
        optionMenu()

    return choice == "b"


def scale_image(img, width, height):
    height_ori, width_ori, _ = img.shape
   
    scale_h = height/height_ori
    scale_w  = width/width_ori

    scale = min(scale_h, scale_w)

    width = int( width_ori * scale)
    height = int (height_ori * scale)
    print("Original size:", width_ori, height_ori)
    print("New size:", width, height)
    return cv.resize(img, (width, height), interpolation = cv.INTER_LINEAR)

def circular_section(img, circle):

    x,y,r = circle
    mask = np.zeros_like(img)
    cv.circle(mask, (x, y), r, (255, 255, 255), cv.FILLED)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]

    return out

def main(img):
    # Ask if Circle Hough Transform is to be chosen
    useHough = optionMenu()

    img = scale_image(img, IMAGE_SCALE_WIDTH, IMAGE_SCALE_HEIGHT)

    # Show image
    cv.imshow("Img", img)

    # Smooth image maintaing edges
    bgr = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    img = cv.bilateralFilter(bgr, 10, 30, 30)

    # Remove small noise from edges left from bilateral filter
    img = cv.medianBlur(img, 5)

    # Equalize saturation
    img_eq = saturation_eq(img)

    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # h, s, v = cv.split(hsv)
    # quartil1 = np.quantile(s, .75)
    # quartil2 = np.quantile(s, .95)
    # img_eq = stretch_sat(img, quartil1, quartil2)


    # hsv segmentation
    masks = [
        hsvRedSegmentation(img_eq),
        hsvBlueSegmentation(img_eq)
    ]
    name = [
        "Red",
        "Blue"
    ]

    for i, mask in enumerate(masks):

        height, width = mask.shape

        # Remove small noise
        mask = cv.medianBlur(mask, 3)

        # Increase border to prevent dilate with canvas edges
        border_size = 50
        mask = cv.copyMakeBorder(
            mask, border_size, border_size, border_size, border_size, cv.BORDER_CONSTANT)

        # Close mask gaps
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # Remove border
        mask = mask[border_size:height + border_size,
                    border_size:width + border_size]

        masks[i] = mask
        cv.imshow('Mask' + str(i), masks[i])

    # Edge canvas
    height, width, channels = img.shape
    edge_canvas = np.zeros((height, width, 1), np.uint8)

    # Find circles with Hough Transform, if chosen
    if (useHough):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        circles = getCircleCountours(v)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            i = 0
            for circle in circles:
                i+=1
                section = circular_section(img, circle)
                section = saturation_eq(section)
                mask  = hsvRedSegmentation(section)

            for (x, y, r) in circles:
                cv.circle(img, (x, y), r, (255, 0, 0), 3)
                cv.circle(img, (x, y), 2, (255, 0, 255), 3)
                cv.putText(img, 'Circle', (x, y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for i, mask in enumerate(masks):

        # find countours
        contours = getContours(mask)

        for cnt in contours:

            # close open contours
            hull = cv.convexHull(cnt)

            # discard noise
            if cv.contourArea(hull) < 100:
                continue

            shape_name, shape = identifyShape(hull)
            x, y = center_cnt(hull)

            if (shape_name == "Circle" and useHough):
                continue

            cv.drawContours(edge_canvas, [hull], -1, (255, 255, 255), 1)
            cv.drawContours(img, [shape], -1, (0, 255, 0), 2)
            cv.putText(img, name[i] + ' ' + shape_name, (x - 40, y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv.imshow('Edges', edge_canvas)
    cv.imshow('ImgEq', img_eq)
    cv.imshow('Final', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--f', help="The image filepath")

    img = None
    file = parser.parse_args().f

    if (file):
        img = load_image(file)
    else:
        img = capture_frame()

    main(img)
