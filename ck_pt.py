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
    import sys
    import cv2
    from glob import glob
    # set test image path
    image_paths = glob('test_images/test*.jpg')

    # prepare the objs for landfinder
    ## create calibrator with parameter file
    calibrator = Calibrator('calibration.p')
    ## create perspective transformer
    ptransformer = PTransformer(src=src, dst=dst)

    col = 2
    row = len(image_paths)
    fig = plt.figure(figsize=(5.*col, 4.*row))

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_image = image.copy()
        orig_image = cv2.polylines(orig_image, np.int_([src]), isClosed=True, color=(40, 40, 250), thickness = 10)
        warp = ptransformer.transform(image)
        num = 2*idx + 1
        ax1 = fig.add_subplot(row, col, num)
        ax1.set_title("test{0}_original".format(idx + 1))
        ax1.imshow(orig_image)
        ax1.axis('off')
        ax2 = fig.add_subplot(row, col, num + 1)
        ax2.set_title("test{0}_warp".format(idx + 1))
        ax2.imshow(warp)
        ax2.axis('off')
    fig.tight_layout()
    plt.savefig('output_images/perspective_transform_results.png')
    plt.show()