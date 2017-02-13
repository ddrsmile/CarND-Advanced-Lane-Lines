# CarND-Advanced-Lane-Finding

_**Nan-Tsou Liu**_ 2017-02-13

[![project_video_result](output_images/project_video_result.gif)](https://youtu.be/iCOmGSDvGCc)

click for **youtube** videos

## Abstract

### Intrdouction

_**Advanced Lane Finding**_ is one of _**Udacity**_ **Self-Driving Car Engineer Nanodegree**. The task is to correctly detect the lane lines on the road and to draw the overlay on it. The resource of the images is the video provided by _**Udacity**_. The road are recoded by a camera set to the front of a car. Besides, the iamges of chessboard and 6 images for testing are also provided.

--
### Approach & Result

As the approach of this project, I built the models cotains following methodologies, `camera calibration`, `perspective transform`, `Color thresholding` and `polynomial fitting`. Fist of all, `camera calibration` was carried out with the given chessboard images. I defined the source (`src`) area which contains the lane lines I want to found by the models and the destination (`dst`) which is used to `perspective transform`. As what instruction suggested, I warpped the images into bird's-eye view. As the third step, I applied `Color thresholding` only to extract **yellow** and **white** color. After lots of trial and error, I eventually used **L** channel of **LUV**, **b** channel of **Lab**, **yellow** area of **HVS** and **white** area of **HLS**. I did not use `Sobel thresholding` because I was satisfied the results of `Color thresholding`. 

And the goal is to correctly detect the lane lines on the road by the skills of computer vision and then to draw the overlay well road is captured by the camera set on the front of a car. _**CV2**_ is the main module used to preprocess the images and _**numpy**_ is mainly used to operate the dataset. Besides, The concepts like  are applied in the task to complete the project. As the result shown below, The lane lines are suceefully found and the overlay is perfectly drawn on the road.

## Approach

### Calibration

--

### Perspective Transform

--

### Color Mask

--

### Lane Finding

--

### Fit Polynomial

--

### Radius of Curvature and Position in Lane

--

### Drawing Result

--

## Reflection
