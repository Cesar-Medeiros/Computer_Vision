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

IMAGES_DIR = 'images/'
IMAGES_SIMPLE_DIR = IMAGES_DIR + 'simple/'
IMAGES_COMPLEX_DIR = IMAGES_DIR + 'complex/'
IMAGES_OTHER_DIR = IMAGES_DIR + 'other/'


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

def hsvRedSegmentation(img):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

    # first range of red
    lower_red = np.array([0,200,50])
    upper_red = np.array([5,255,255])
    mask1 = cv.inRange(hsv,lower_red,upper_red)

    # second range of red
    lower_red = np.array([170,200,50])
    upper_red = np.array([180,255,255])
    mask2 = cv.inRange(hsv,lower_red,upper_red)

    # combining masks
    mask = mask1 + mask2
    return mask

def hsvBlueSegmentation(img):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    lower_blue = np.array([104,200,100])
    upper_blue = np.array([115,255,255])
    mask = cv.inRange(hsv,lower_blue,upper_blue)
    return mask

def getEdges(gray):
    return cv.Canny(gray,threshold1=50,threshold2=200)

def getContours(edges):
    contours,_ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def getApproxContour(contour):
    # between 1% and 5%
    epsilon = 0.04*cv.arcLength(cnt,True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    return approx

def getShapeName(numVertices):
    text = ""
    if numVertices == 3:
        text = "Triangle"
    elif numVertices == 4:
        text = "Rectangle"
    elif numVertices == 8:
        text = "Octogon"
    else:
        text = "Circle"

    return text

def saturation_eq(img):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    new_s = cv.equalizeHist(s)
    hsv_eq = cv.merge([h, new_s, v])
    bgr_eq = cv.cvtColor(hsv_eq,cv.COLOR_HSV2BGR)
    return bgr_eq

def increase_sat(img, alpha, beta):
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    new_s = cv.convertScaleAbs(s, alpha=alpha, beta=beta)
    hsv_eq = cv.merge([h, new_s, v])
    bgr_eq = cv.cvtColor(hsv_eq,cv.COLOR_HSV2BGR)
    return bgr_eq

def test(img1):
    # Create zeros array to store the stretched image
    minmax_img = np.zeros((img1.shape[0],img1.shape[1]),dtype = 'uint8')
    
    # Loop over the image and apply Min-Max formulae
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            minmax_img[i,j] = 255*(img1[i,j]-np.min(img1))/(np.max(img1)-np.min(img1))
    
    return minmax_img


if __name__ == '__main__':
    print(__doc__)

    # load image
    # img = capture_frame()
    img = load_image(IMAGES_COMPLEX_DIR + '1.jpg')
    cv.imshow("Img", img)

    # Smooth image maintaing edges
    img = cv.bilateralFilter(img,10,30,30)

    # Remove small noise from edges left from bilateral filter
    img = cv.GaussianBlur(img, (3,3), 0)

    #Equalize saturation
    # img_eq = increase_sat(img, 1, 20)
    img_eq = saturation_eq(img)

    cv.imshow('ImgEq',img_eq)
   


    # hsv segmentation
    masks = [
                hsvRedSegmentation(img_eq),
                hsvBlueSegmentation(img_eq)
            ]
    name = [
        "red",
        "blue"
    ]

    for i, mask in enumerate(masks):

        # Remove small noise
        mask = cv.medianBlur(mask, 3)

        # Close mask gaps
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        cv.imshow('Mask' + str(i),mask)


    # Edge canvas
    height, width, channels = img.shape 
    edge_canvas = np.zeros((height,width,1), np.uint8)

    for i, mask in enumerate(masks):

        # find countours
        contours = getContours(mask)

        for cnt in contours:
            # close open contours
            hull = cv.convexHull(cnt)

            # discard noise
            if cv.contourArea(hull) < 100:
                cv.fillPoly(mask, pts =[cnt], color=(0,0,0))
                continue

            approx = getApproxContour(cnt)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            shapeName = getShapeName(len(approx))

            cv.drawContours(edge_canvas, [cnt], -1, (255, 255, 255), 1)
            cv.drawContours(img, [approx], -1, (0, 255, 0), 2)
            cv.putText(img, name[i] + ' ' + shapeName + ' ' + str(len(approx)), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    

    cv.imshow('Edges',edge_canvas)
    cv.imshow('Final', img)
    cv.waitKey(0)
    cv.destroyAllWindows()