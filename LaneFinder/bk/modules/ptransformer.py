# -*- coding: utf-8 -*-
# import required modules
import cv2
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

if __name__=='__main__':
    import numpy as np
    from calibrator import Calibrator
    import matplotlib.pyplot as plt

    calibrator = Calibrator('../calibration.p')

    OFFSET = 250
    src = np.float32([
            (132, 703),
            (540, 466),
            (740, 466),
            (1147, 703)])
    dst = np.float32([
        (src[0][0] + OFFSET, 720),
        (src[0][0] + OFFSET, 0),
        (src[-1][0] - OFFSET, 0),
        (src[-1][0] - OFFSET, 720)])

    src = np.float32([[490, 480],[810, 480],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    
    ptransformer = PTransformer(src=src, dst=dst)

    image = cv2.imread('../../test_images/test1.jpg')
    undist = calibrator.undistort(image)
    warped = ptransformer.transform(undist)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    ax1.set_title('Undistorted Image', fontsize=20)
    ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted and Warped Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()