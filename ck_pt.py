# -*- coding: utf-8 -*-
# import required modules
import numpy as np

from LaneFinder.calibrator import Calibrator
from LaneFinder.ptransformer import PTransformer
from LaneFinder.masker import Masker
from LaneFinder.lanefinder import LaneFinder

# define the windows for the perspective transform
src = np.float32([
    [595, 450],
    [690, 450],
    [1115, 720],
    [216, 720]
])

dst = np.float32([
    [450, 0],
    [830, 0],
    [830, 720],
    [450, 720]
])

def multi_enumerate(list, step):
    if len(list)%step != 0:
        raise Exception('the length of list should be a multiple of step')
    for i in range(0, len(list), steps):
        yield i, list[i], i+1, list[i+1]
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import sys
    import cv2
    from glob import glob
    # set test image path
    #image_paths = glob('test_images/test*.jpg')
    image_paths = ['test_images/test3.jpg','test_images/test4.jpg','test_images/test5.jpg']

    # prepare the objs for landfinder
    ## create calibrator with parameter file
    calibrator = Calibrator('calibration.p')
    ## create perspective transformer
    ptransformer = PTransformer(src=src, dst=dst)

    col = 2
    row = len(image_paths)
    #fig = plt.figure(figsize=(5.*col, 3.5*row))
    fig = plt.figure()
    gs1 = gridspec.GridSpec(row, col)
    gs1.update(wspace=0., hspace=0.)

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = calibrator.undistort(image)
        orig_image = image.copy()
        orig_image = cv2.polylines(orig_image, np.int_([src]), isClosed=True, color=(40, 40, 250), thickness = 5)
        warp = ptransformer.transform(image)

        num = 2*idx
        ax1 = plt.subplot(gs1[num])
        #ax1 = fig.add_subplot(row, col, num)
        ax1.set_frame_on(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(orig_image)
        ax1.set_aspect('auto')
        ax1.title.set_visible(False)
        ax1.axis('off')
        #ax2 = fig.add_subplot(row, col, num + 1)
        ax2 = plt.subplot(gs1[num + 1])
        ax2.set_frame_on(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(warp)
        ax2.set_aspect('auto')
        ax2.title.set_visible(False)
        ax2.axis('off')
    plt.savefig('output_images/perspective_transform_results.png',
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)
    #plt.show()