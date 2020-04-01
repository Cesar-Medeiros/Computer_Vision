#!/usr/bin/env python

'''
Signal Recognition

Usage:
   vcom_p1.py
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

font = cv.FONT_HERSHEY_COMPLEX

IMAGE_FN = 'image.jpeg'
IMAGE2_FN = 'image2.jpg'
IMAGE3_FN = 'image3.jpg'
IMAGE4_FN = 'image4.png'

def load_image(fn):
    fn = cv.samples.findFile(fn)
    print('loading "%s" ...' % fn)
    img = cv.imread(fn, cv.IMREAD_UNCHANGED)
    return img

def capture_frame():
    cap = cv.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        cv.imshow('Webcam', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            return frame

def hsvSegmentation(img):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    # first range of red
    lower_red = np.array([0,150,100])
    upper_red = np.array([10,255,255])
    mask1 = cv.inRange(hsv,lower_red,upper_red)
    # second range of red
    lower_red = np.array([170,150,100])
    upper_red = np.array([180,255,255])
    mask2 = cv.inRange(hsv,lower_red,upper_red)
    # combining masks
    mask = mask1 + mask2
    maskN = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3),np.uint8))
    maskN = cv.morphologyEx(mask, cv.MORPH_DILATE, np.ones((3,3),np.uint8))
    return maskN

def getEdges(gray):
    return cv.Canny(gray,threshold1=50,threshold2=200)

def getContours(edges):
    contours,_ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def getApproxContour(contour):
        epsilon = 0.04*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        return approx

def getShapeName(numVertices):
    text = ""
    if numVertices == 3:
        text = "Triangle"
    elif numVertices == 4:
        text = "Rectangle"
    elif numVertices == 5:
        text = "Pentagon"
    elif numVertices == 6:
        text = "Hexagon"
    else:
        text = "Circle"

    return text

if __name__ == '__main__':
    print(__doc__)

    # load image
    # img = capture_frame()
    img = load_image(IMAGE3_FN)


    # hsv red segmentation
    mask = hsvSegmentation(img)

    gray = mask
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]

    # canny edge detection
    edges = getEdges(thresh)
    cv.imshow('Edges',edges)

    # find countours
    contours = getContours(edges)

    contoursImage = img.copy()
    # cv.drawContours(contoursImage, contours, -1, (0, 255, 255), 2)

    # identify shape
    for cnt in contours:
        if cv.contourArea(cnt) < 100:
            continue
        approx = getApproxContour(cnt)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        shapeName = getShapeName(len(approx))
        cv.drawContours(contoursImage, [cnt], -1, (0, 255, 255), 2)
        cv.putText(contoursImage, shapeName, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # draw countours
    cv.imshow('Countours',contoursImage)

    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()