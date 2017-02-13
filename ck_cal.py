# -*- coding: utf-8 -*-
# import required modules
import numpy as np
from LaneFinder.calibrator import Calibrator

# define the windows for the perspective transform
src = np.float32([
    [590, 450],
    [720, 450],
    [1115, 720],
    [170, 720]
])

dst = np.float32([
    [450, 0],
    [850, 0],
    [850, 720],
    [450, 720]
])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    import cv2
    from glob import glob
    import random

    # set chessboard images' path
    image_paths = glob('camera_cal/calibration*.jpg')
    # create calibrator
    calibrator = Calibrator()
    # calibrate camera with chessboard images without exporting the result
    calibrator.calibrate(image_paths=image_paths, export=False)

    ## because not every image's corners can be found
    ## set the count to drop out the show case loop
    image = cv2.imread('camera_cal/calibration2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_image = image.copy()
    # get corner of the chessboard
    ret, corners = calibrator.get_corners(image)
    # put the corner on chessboard image if ret found
    if ret:
        cv2.drawChessboardCorners(image, (9, 6), corners, ret)
        undist = calibrator.undistort(image)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.set_frame_on(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('original image')
        ax1.imshow(orig_image)
        ax1.axis('off')
        ax2.set_frame_on(False)
        ax2.set_xticks([]) 
        ax2.set_yticks([])
        ax2.set_title('undistorted image with corners')
        ax2.imshow(undist)
        ax2.axis('off')

        plt.savefig('output_images/calibrated_results.png',
                    bbox_inches='tight',
                    transparent="True", 
                    pad_inches=0)
        plt.show()
    
    image_paths = glob('test_images/test4.jpg')

    col = 2
    row = len(image_paths)
    fig = plt.figure(figsize=(5.*col, 3.5*row))
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        num = 2*idx + 1
        ax1 = fig.add_subplot(row, col, num)
        ax1.set_title("test{}_original".format(idx + 1))
        ax1.axis('off')
        ax1.set_frame_on(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(image)
        # apply calibrator.undist on image
        undist = calibrator.undistort(image)
        ax2 = fig.add_subplot(row, col, num+1)
        ax2.set_title("test{}_undistorted".format(idx + 1))
        ax2.axis('off')
        ax2.set_frame_on(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(undist)

    plt.savefig('output_images/calibrator_results.png',
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)
    plt.show()