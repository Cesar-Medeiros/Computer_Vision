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

IMAGE_FN = 'image.jpeg'
IMAGE2_FN = 'image2.jpg'

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
    lower_red = np.array([0,150,0])
    upper_red = np.array([10,255,255])
    mask1 = cv.inRange(hsv,lower_red,upper_red)
    # second range of red
    lower_red = np.array([170,150,0])
    upper_red = np.array([180,255,255])
    mask2 = cv.inRange(hsv,lower_red,upper_red)
    # combining masks
    mask = mask1 + mask2
    mask1 = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv.morphologyEx(mask, cv.MORPH_DILATE, np.ones((3,3),np.uint8))
    return cv.bitwise_and(img,img,mask=mask1)

def getEdges(hsv):
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    gray = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
    return cv.Canny(gray,threshold1=50,threshold2=200)

def getContours(img,edges):
    contours,_ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return cv.drawContours(img, contours, -1, (0, 255, 0), 2)

if __name__ == '__main__':
    print(__doc__)

    # load image
    # img = capture_frame()
    img = load_image(IMAGE2_FN)
    cv.imshow('Signal', img)

    # hsv red segmentation
    hsv = hsvSegmentation(img)
    cv.imshow('HSV Segmentation',hsv)

    # canny edge detection
    edges = getEdges(hsv)
    cv.imshow('Edges',edges)

    # find countours
    contours = getContours(img,edges)
    cv.imshow('Contours',contours)

    cv.waitKey(0)
    cv.destroyAllWindows()