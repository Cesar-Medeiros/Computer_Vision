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


def load_image(fn):
    fn = cv.samples.findFile(fn)
    print('loading "%s" ...' % fn)
    img = cv.imread(fn, cv.IMREAD_UNCHANGED)
    return img

if __name__ == '__main__':
    print(__doc__)

    img = load_image(IMAGE_FN)
    cv.imshow('Signal', img)

    cv.waitKey(0)
    cv.destroyAllWindows()

