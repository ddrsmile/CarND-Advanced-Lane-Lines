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
    [830, 0],
    [830, 720],
    [450, 720]
])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
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
    fig = plt.figure()
    gs1 = gridspec.GridSpec(row, col)
    gs1.update(wspace=0., hspace=0.)

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = lanefinder.process(image)
        ax = plt.subplot(gs1[idx])
        ax.title.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.imshow(image)
        ax.set_aspect('auto')
    plt.savefig('output_images/lanefinder_results.png',
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)
    plt.show()