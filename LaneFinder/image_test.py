# -*- coding: utf-8 -*-
from models import *
#from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip

color_models = [
    #cv2.COLOR_RGB2HLS,
    cv2.COLOR_RGB2LUV,
    cv2.COLOR_RGB2Lab
]

#nth_chs = [2, 0, 2]
nth_chs = [ 0, 2]
thresholds = [
        #(180, 255),
        (215, 255),
        (145, 200)
    ]

src = np.float32([[490, 482],[810, 482],
                    [1250, 720],[0, 720]])
dst = np.float32([[0, 0], [1280, 0], 
                    [1250, 720],[40, 720]])

'''
OFFSET = 0
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
'''
#dst = np.float32([
#    (src[0][0] + OFFSET, 720),
#    (src[0][0] + OFFSET, 0),
#    (src[-1][0] - OFFSET, 0),
#    (src[-1][0] - OFFSET, 720)])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    #image = cv2.imread('../challenge_images/frame240.jpg')
    imgpath = sys.argv[1]
    image = cv2.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    origin = image.copy()
    calibrator = Calibrator('../calibration.p')
    undist = calibrator.undistort(image)
    ptransformer = PTransformer(src=src, dst=dst)
    warp = ptransformer.transform(undist)
    masker = Masker(color_models=color_models, nth_chs=nth_chs, thresholds=thresholds)
    masked = masker.get_masked_image(warp)
    #yellow = masker.extract_yellow(image)
    #gray = masker.grayscale(image)
    #lanefinder = LaneFinder(calibrator=calibrator, ptransformer=ptransformer, masker=masker, scan_image_steps=7, margin=25)
    #masked = ptransformer.transform(masked)
    #image = lanefinder.process(image)
    #plt.imshow(image)
    #plt.imshow(cv2.cvtColor(wrap, cv2.COLOR_BGR2RGB))
    #kk = np.zeros_like(yellow)

    #kk[(yellow == 1)] = 1

    f, (ax1, ax2) = plt.subplots(1,2, figsize=(9, 6))
    f.tight_layout()

    ax1.imshow(warp)
    ax2.imshow(masked, cmap='gray')

    plt.show()