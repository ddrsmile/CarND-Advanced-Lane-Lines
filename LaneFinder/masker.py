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

class Masker(object):
    def __init__(self, user_default=True):
        self.user_default = user_default

    def build_default_binaries(self, image):
        binaries = []
        binaries.append(self.extract_yellow(image))
        binaries.append(self.extract_white(image))
        binaries.append(self.extract_l_of_luv(image))
        binaries.append(self.extract_b_of_lab(image))
        return binaries

    def get_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def get_lab(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    def get_hls(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    def get_luv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

    def apply_threshold(self, channel, thresh_min=0, thresh_max=255):
        binary = np.zeros_like(channel)
        binary[(channel >= thresh_min) & (channel <= thresh_max)] = 1
        return binary

    def set_channels(self, warpped_image, color_models=None, nth_chs=None):
        channels = []
        # use self.color_models if color_models is not given
        color_models = color_models
        # use self.nth_elements if nth_elements is not given
        nth_chs = nth_chs or self.nth_chs
        for model, nth_ch in zip(color_models, nth_chs):
            channels.append(cv2.cvtColor(warpped_image, model)[:,:,nth_ch])
        return channels

    def build_binary_with_thresholds(self, channels=None, thresholds=None):
        thresholds = thresholds
        binaries = []
        for channel, threshold in zip(channels, thresholds):
            binaries.append(self.apply_threshold(channel, threshold[0], threshold[1]))
        return binaries

    def combine_binaries(self, binaries):
        if binaries is None or len(binaries) == 0:
            return None
        combined_binary = np.zeros_like(binaries[0])
        for binary in binaries:
            combined_binary = cv2.bitwise_or(combined_binary, binary)
        return combined_binary

    def get_masked_image(self, image, other_binaries=None):

        if self.user_default:
            binaries = self.build_default_binaries(image)
        else:
            binaries = []
        
        if other_binaries is not None:
            binaries.extend(other_binaries)

        combined_binary = self.combine_binaries(binaries)
        return combined_binary

    def extract_l_of_luv(self, image):
        luv = self.get_luv(image)
        l = luv[:, :, 0]
        return self.apply_threshold(l, 215, 255)

    def extract_b_of_lab(self, image):
        lab = self.get_lab(image)
        b = lab[:, :, 2]
        return self.apply_threshold(b, 145, 200)

    def extract_yellow(self, image):
        hsv = self.get_hsv(image)
        yellow = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))//255
        return yellow

    def extract_white(self, image):
        hls = self.get_hls(image)
        white = cv2.inRange(hls, (0, 206, 0), (180, 255, 255))//255
        return white