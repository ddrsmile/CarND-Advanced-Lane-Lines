# -*- coding: utf-8 -*-
# import required modules
import cv2
import numpy as np
import pickle

"""
cv2.anything() --> use (width, height)
image.anything() --> use (height, width)
numpy.anything() --> use (height, width)
"""

class PTransformer(object):
    def __init__(self, src=None, dst=None):
        self.src = src
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.inv_M = cv2.getPerspectiveTransform(dst, src)
    
    def set_src(self, src):
        self.src = src
    
    def set_dst(self, dst):
        self.dst = dst
    
    def transform(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    def inv_transform(self, image):
        return cv2.warpPerspective(image, self.inv_M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)