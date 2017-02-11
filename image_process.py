# -*- coding: utf-8 -*-
from models import *

'''
src = np.float32([[490, 480],[810, 480],
                    [1250, 720],[40, 720]])
dst = np.float32([[0, 0], [1280, 0], 
                    [1280, 720],[0, 720]])
'''
src = np.float32([
    [590, 450],
    [720, 450],
    [1115, 720],
    [170, 720],
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
    #image = cv2.imread('../challenge_images/frame240.jpg')
    imgpath = sys.argv[1]
    image = cv2.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    origin = image.copy()
    calibrator = Calibrator('../calibration.p')
    #undist = calibrator.undistort(image)
    ptransformer = PTransformer(src=src, dst=dst)
    #warp = ptransformer.transform(undist)
    masker = Masker()
    #masked = masker.get_masked_image(warp)
    #yellow = masker.extract_yellow(image)
    #gray = masker.grayscale(image)
    lanefinder = LaneFinder(calibrator=calibrator, ptransformer=ptransformer, masker=masker, scan_image_steps=7, margin=25)
    #masked = ptransformer.transform(masked)
    image = lanefinder.process(image)
    #plt.imshow(image)
    #plt.imshow(cv2.cvtColor(wrap, cv2.COLOR_BGR2RGB))
    #kk = np.zeros_like(yellow)

    #kk[(yellow == 1)] = 1

    f, (ax1) = plt.subplots(1,1, figsize=(9, 6))
    f.tight_layout()

    ax1.imshow(image)
    #ax2.imshow(masked, cmap='gray')

    plt.show()