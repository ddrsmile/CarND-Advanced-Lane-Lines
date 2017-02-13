# -*- coding: utf-8 -*-
# import required modules
from LaneFinder.calibrator import Calibrator
from LaneFinder.ptransformer import PTransformer
from LaneFinder.masker import Masker
from LaneFinder.lanefinder import LaneFinder

titles = ['original', 'combined', 'yellow', 'white', 'l of luv', 'b of lab']

def create_image_set(masker, image):
    image_set = dict()
    image_set['original'] = image.copy()
    image_set['combined'] = masker.get_masked_image(image.copy())
    image_set['yellow'] = masker.extract_yellow(image.copy())
    image_set['white'] = masker.extract_white(image.copy())
    image_set['l of luv'] = masker.extract_l_of_luv(image.copy())
    image_set['b of lab'] = masker.extract_b_of_lab(image.copy())

    return image_set




if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import cv2
    from glob import glob
    import random
    # set test image path
    image_paths = glob('test_images/test*.jpg')
    # randomly get the images to show the result
    samples = random.sample(image_paths, 3)
    # create masker
    masker = Masker()

    for sidx, sample in enumerate(samples):
        image = cv2.imread(sample)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_set = create_image_set(masker, image)

        row = 3
        col = len(titles) // row if len(titles) % row == 0 else len(titles) // row + 1
        fig = plt.figure()
        gs1 = gridspec.GridSpec(row, col)
        gs1.update(wspace=0.05, hspace=0.2)
        for idx, title in enumerate(titles):
            num = idx + 1
            #ax = fig.add_subplot(row, col, num)
            ax = plt.subplot(gs1[idx])
            ax.set_title(title)
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            if title != 'original':
                ax.imshow(image_set[title], cmap='gray')
            else:
                ax.imshow(image_set[title])
            ax.set_aspect('auto')

        plt.savefig('output_images/masker_results_{}.png'.format(sidx+1),
                    bbox_inches='tight',
                    transparent="True", 
                    pad_inches=0)
        #plt.show()