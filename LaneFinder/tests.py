# -*- coding: utf-8 -*-
from models import *
#from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip

color_models = [
    #cv2.COLOR_BGR2HLS,
    cv2.COLOR_BGR2LUV,
    cv2.COLOR_BGR2Lab
]

#nth_elements = [2, 0, 2]
nth_elements = [0, 2]

thresholds = [
        #(180, 255),
        (215, 255),
        (145, 200)
    ]

src = np.float32([[490, 480],[810, 480],
                    [1250, 720],[40, 720]])
dst = np.float32([[0, 0], [1280, 0], 
                    [1250, 720],[40, 720]])


VIDEOS = ["../videos/project_video.mp4", '../videos/challenge_video.mp4']
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = cv2.imread('../test_images/test5.jpg')
    calibrator = Calibrator('../calibration.p')
    ptransformer = PTransformer(src=src, dst=dst)
    #image = ptransformer.transform(image)
    masker = Masker(color_models=color_models, nth_elements=nth_elements, thresholds=thresholds)
    #image = masker.get_masked_image(image)
    lanefinder = LaneFinder(calibrator=calibrator, ptransformer=ptransformer, masker=masker, n_image=7)

    clip1 = VideoFileClip(VIDEOS[1])
    project_clip = clip1.fl_image(lanefinder.process)
    project_output = VIDEOS[1][:-4] + '_test.mp4'
    project_clip.write_videofile(project_output, audio=False)

    #image = lanefinder.process(image)
    #image = lanefinder.test_image
    #plt.imshow(image, cmap='gray')
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.show()