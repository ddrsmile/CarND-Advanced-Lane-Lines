# -*- coding: utf-8 -*-
# import required modules
import cv2
import numpy as np

class Masker(object):
    def __init__(self):
        self.binaries = []
    
    def apply_sobel(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # create a binary image of ones where threshold is met, zeros otherwise
        binary = np.zeros_like(gradmag)
        binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        self.binaries.append(binary)


    def extract_channel(self, channel, thresh_min=0, thresh_max=255):
        binary = np.zeros_like(channel)
        binary[(channel >= thresh_min) & (channel <= thresh_max)] = 1
        self.binaries.append(binary)
    
    def combine_binary(self, binaries=None):
        binaries = binaries or self.binaries
        combined_binary = np.zeros_like(binaries[0])
        for binary in binaries:
            combined_binary[(combined_binary == 1) | (binary == 1)] = 1
        return combined_binary
    
    def apply_channel_threshold(self, channels, thresholds):
        for ch, thres in zip(channels, thresholds):
            self.extract_channel(ch, thres[0], thres[1])
        



    def get_binaries(self):
        return self.binaries


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from calibrator import Calibrator
    from ptransformer import PTransformer

    calibrator = Calibrator('../scripts/calibration.p')
    src = np.float32([[490, 480],[810, 480],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    
    ptransformer = PTransformer(src=src, dst=dst)

    image = cv2.imread('../test_images/test5.jpg')
    undist = calibrator.undistort(image)
    warped = ptransformer.transform(undist)

    masker = Masker()

    channels = [
        cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)[:,:,2],
        cv2.cvtColor(warped, cv2.COLOR_BGR2LUV)[:,:,0],
        cv2.cvtColor(warped, cv2.COLOR_BGR2Lab)[:,:,2]
    ]

    thresholds = [
        (180, 255),
        (225, 255),
        (155, 200)
    ]

    masker.apply_channel_threshold(channels=channels, thresholds=thresholds)

    #masker.apply_threshold(channel=s_channel, thresh_min=180, thresh_max=255)
    #masker.apply_threshold(channel=l_channel, thresh_min=225, thresh_max=255)
    #masker.apply_threshold(channel=b_channel, thresh_min=155, thresh_max=200)

    #masker.apply_sobel(warped, mag_thresh=(100, 200))

    binaries = masker.get_binaries()

    combined_binary = masker.combine_binary()

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey='col', sharex='row', figsize=(10,4))
    f.tight_layout()
    
    ax1.set_title('Original Image', fontsize=16)
    ax1.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    
    ax2.set_title('Warped Image', fontsize=16)
    ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype('uint8'))
    
    ax3.set_title('Combined color thresholds', fontsize=16)
    ax3.imshow(combined_binary, cmap='gray')


    ax4.set_title('S of HLS', fontsize=16)
    ax4.imshow(binaries[0], cmap='gray')

    ax5.set_title('L of LUV', fontsize=16)
    ax5.imshow(binaries[1], cmap='gray')
    
    ax6.set_title('B of Lab', fontsize=16)
    ax6.imshow(binaries[2], cmap='gray')

    plt.show()