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
    import sys
    from moviepy.editor import VideoFileClip

    calibrator = Calibrator('calibration.p')
    ptransformer = PTransformer(src=src, dst=dst)
    masker = Masker()
    lanefinder = LaneFinder(calibrator=calibrator, ptransformer=ptransformer, 
                            masker=masker, n_image=5, scan_image_steps=10, margin=25)

    video_path = 'videos/project_video.mp4'
    clip1 = VideoFileClip(video_path)
    project_clip = clip1.fl_image(lanefinder.process)
    project_output = video_path[:-4] + '_result.mp4'
    project_clip.write_videofile(project_output, audio=False)