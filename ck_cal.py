# -*- coding: utf-8 -*-
# import required modules
import numpy as np

from LaneFinder.calibrator import Calibrator
from LaneFinder.ptransformer import PTransformer
from LaneFinder.masker import Masker
from LaneFinder.lanefinder import LaneFinder

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
    # set test image path
    image_paths = glob('test_images/test*.jpg')

    # prepare the objs for landfinder
    ## create calibrator with parameter file
    calibrator = Calibrator('calibration.p')
    ## create perspective transformer
    ptransformer = PTransformer(src=src, dst=dst)
    ## create masker
    masker = Masker()
    # create LaneFinder
    ## set n_image to 1 because it is used for images
    lanefinder = LaneFinder(calibrator=calibrator, ptransformer=ptransformer, 
                            masker=masker, n_image=1, scan_image_steps=10, margin=25)
    col = 2
    row = len(image_paths)//2 if len(image_paths) % 2 == 0 else len(image_paths)//2 + 1
    fig = plt.figure(figsize=(5.*col, 5.*row))


    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = lanefinder.process(image)
        num = idx + 1
        ax = fig.add_subplot(row, col, num)
        ax.set_title("test{}".format(num))
        ax.imshow(image)
    fig.tight_layout()
    plt.show()