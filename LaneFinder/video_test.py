# -*- coding: utf-8 -*-
from models import *
#from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip
'''
color_models = [
    cv2.COLOR_RGB2HLS,
    cv2.COLOR_RGB2LUV,
    cv2.COLOR_RGB2Lab
]

nth_chs = [2, 0, 2]

thresholds = [
        (180, 255),
        (215, 255),
        (145, 200)
    ]
'''
color_models = [
    cv2.COLOR_RGB2LUV,
    cv2.COLOR_RGB2Lab
]

nth_chs = [0, 2]

thresholds = [
        (220, 255),
        (170, 255)
]


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
src = np.float32([[490, 480],[810, 480],
                    [1250, 720],[40, 720]])
dst = np.float32([[0, 0], [1280, 0], 
                    [1280, 720],[0, 720]])

VIDEOS = ["../videos/project_video.mp4", '../videos/challenge_video.mp4', '../videos/harder_challenge_video.mp4']
if __name__ == '__main__':
    import sys
    calibrator = Calibrator('../calibration.p')
    ptransformer = PTransformer(src=src, dst=dst)
    masker = Masker()
    lanefinder = LaneFinder(calibrator=calibrator, ptransformer=ptransformer, masker=masker, n_image=5, scan_image_steps=10, margin=25)

    video_path = sys.argv[1]

    clip1 = VideoFileClip(video_path)
    project_clip = clip1.fl_image(lanefinder.process)
    project_output = video_path[:-4] + '_test.mp4'
    project_clip.write_videofile(project_output, audio=False)