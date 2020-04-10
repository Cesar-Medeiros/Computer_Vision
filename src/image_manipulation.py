import cv2 as cv
import numpy as np

def scale_image(img, width, height):
    height_ori, width_ori, _ = img.shape
   
    scale_h = height/height_ori
    scale_w  = width/width_ori

    scale = min(scale_h, scale_w)

    width = int( width_ori * scale)
    height = int (height_ori * scale)
    return cv.resize(img, (width, height), interpolation = cv.INTER_LINEAR)


def hist_eq_val(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    new_v = cv.equalizeHist(v)
    hsv_eq = cv.merge([h, s, new_v])
    bgr_eq = cv.cvtColor(hsv_eq, cv.COLOR_HSV2BGR)
    return bgr_eq


def hist_eq_sat(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    new_s = cv.equalizeHist(s)
    hsv_eq = cv.merge([h, new_s, v])
    bgr_eq = cv.cvtColor(hsv_eq, cv.COLOR_HSV2BGR)
    return bgr_eq


def non_linear_sat(img, gamma):
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


def circular_section(img, circle):
    x,y,r = circle
    mask = np.zeros_like(img)
    cv.circle(mask, (x, y), r, (255, 255, 255), cv.FILLED)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]

    return out


def kmeans(img, K):
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2