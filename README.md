# CarND-Advanced-Lane-Finding

_**Nan-Tsou Liu**_ 2017-02-13

## Abstract

### Intrdouction

_**Advanced Lane Finding**_ is one of _**Udacity**_ **Self-Driving Car Engineer Nanodegree**. The task is to correctly detect the lane lines on the road and to draw the overlay on it. The resource of the images is the video provided by _**Udacity**_. The road are recoded by a camera set to the front of a car. Besides, the iamges of chessboard and 6 images for testing are also provided.

--
### Approach

As the approach of this project, I built the models cotains following methodologies, `camera calibration`, `perspective transform`, `Color thresholding` and `polynomial fitting`. The image of the approach is shown below.

![approach](output_images/approach.png)

Fist of all, `camera calibration` was carried out with the given chessboard images. I defined the source (`src`) area which contains the lane lines I wanted to found by the models and the destination (`dst`) which is used to `perspective transform`. As what instruction suggested, I warped the images into bird's-eye view. 

For the third step, I applied `Color thresholding` only to extract **yellow** and **white** color. After lots of trial and error, I eventually used **L** channel of **LUV**, **b** channel of **Lab**, **yellow** area of **HVS** and **white** area of **HLS**. I did not use `Sobel thresholding` because I satisfied the results of `Color thresholding`. Hitsogram was performed to find the positions of the lane lins. Rather than using the half of image as the base, I performed histogram to find the lane lines in each window. After the lane lines are found as the points, I fitted the points with polynomial. And then I calculated the radius of curvature and the position offset with the fitted polynomial functions. 

As the final results, I drew the overlay made by fuond lane lines and put the information of the radius of curvature and position offset on the images.

### Result

[![project_video_result](output_images/project_video_result.gif)](https://youtu.be/QppNfTrM9dY)

click the image for **youtube** videos


## Camera Calibration

**<sub>calibrator.py</sub>**

In this step, I followed what the instruction taught to build my own `calibrator`. `findChessboardCorners`, `calibrateCamera` and `undistort` of _**CV2**_ were mainly used to build the model. `findChessboardCorners` is used to find the corners of the given images of chessboard after the images were grayscaled. And then `calibrateCamera` calculated the camera matrix and distortion coefficients with the corners. In order to resue the calibrated results, I implemented the mehtods to export and loading the results.

As the results shown below, I draw the found corners and showed the undistorted chessboard image to show the difference between distorted image and undistorted image. And then I inputted the testing image given by _**Udacity**_ to check my calibrator.

![calibrated_results](output_images/calibrated_results.png)

![calibrator_results](output_images/calibrator_results.png)

--

### Perspective Transform

**<sub>ptransformer.py</sub>**

`getPerspectiveTransform` was firstly used to calculate `transform matrix M` with defined source, `src` the area to be transformed from, and destination, `dst` the area that we expect how `src` to be transformed. As the suggestion given by the instruction, it is good option that transform `src` into bird's-eye view as `dst`. Besides, I noticed that how `src` and `dst` are defined would influence on the final result, I eventually defined `src` and `dst` as follows:

|src|dst|
|---|---|
|595, 450|450, 0|
|690, 450|830, 0|
|1115, 720|830, 720|
|216, 720|450, 720|

After `transform matrix M` was obtained, `warpPerspective` is used to transform the image into bird's-eye view. And the follows are the example that how perspective transform changes the images.

And in the final step, we have to inverse the transformation for bird's-eye view into its original view. The approach is quite simple that set the `dst` as the **new** `src` and the `src` as the new `dst`. Input new `src` and new `dst` to `getPerspectiveTransform` again. Then we can get a **new** `transform matrix M` as the `inversed transform matrix inv-M`. The results of inversed transform would be shown in very end of this report.

![perspective_transform_results](output_images/perspective_transform_results.png)

--

### Color Mask

**<sub>masker.py</sub>**

At the beginning, I followed the instruction to use `S` channel of **HSV** as `Color thresholding` and **CV2.COLOR_RBG2GRAY** for `Sobel thresholding` to extract the lane lines. The results were good but the overlay is not stable along the video. It would be shifted out of the lane lines at some part, especially for the turing parts.

And therefore I began to investigate **Color models** like `HLS`, `LUV`, `Lab` and so on. Besides, I also noticed that some people using `HSV` or `HLS` to extract yellow and white parts rather than using a channel of a certain color model. So, I was also trying to extract yellow and white parts in the same way. Eventually, I decided to use `L` channel of `LUV` to extract both yellow and white part. And the results of that is not so clear, especially for white, so that I also used `b` channel of `Lab` to extract white parts. Besides, in order to ensure that yellow and white parts can be extracted well, I also applied `HVS` to extract yellow parts and `HLS` to extract white parts.

I followed the way taught by the instruction to apply thresholding, `binary[(channel >= thresh_min) & (channel <= thresh_max)] = 1`, so that output of applying thresholding on single channel is already in binary format. On the other hand, I call `CV2.inRange` to extract yellow and white parts from `HSV` and `HLS` Although the dimension is changed from 3 dimension into 2 dimension, the value of reuslt is 255 rather than 1. So, the postprocess a little bit like normalization is required to make them in binary format.

After all the masks were built with binary format, I combined all these masks as the final output of color mask. And there is figure set to show the result of each mask and final combined mask.

#### Result 1

![masker_results_1](output_images/masker_results_1.png)

#### Result 2

![masker_results_2](output_images/masker_results_2.png)

--

### Lane Finding

**<sub>lanefinder.py: histogram_detection</sub>**

I carried out this methodology by following the instruction. The difference from the way taugh by the instruction is that I took the histogram of each slide rather than taking the histogram of half of the image. But I also implemented a check that using the mean value of the previous slide if there are over 50 points found in the previous slide. By doing this, I can use smaller width of the searching window, the area I check whether there are lane lines. It somehow can reduce the time consuming during lane finding as shown in the code below.

```python
last_base = None
for i in range(steps):
    # define the range in y direction for searching
    end = target_img.shape[0] - (i * px_per_step)
    start = end - px_per_step
    # set last_base to current base if there are more 50 points found in previous image
    if last_base is None:
        # create histogram
        histogram = np.sum(target_img[start:end, :], axis=0)
        # add search_area[0], image offset in x direction, 
        # to ensure the positions of points are correct.
        base = np.argmax(histogram) + search_area[0]
    else:
        base = last_base
    
    # get the indices in the searching area based on "base" and "margin"
    good_inds = self.__get_good_inds(base, margin, start, end)
    # get points in both x and y directions
    cur_x, cur_y = self.nonzerox[good_inds], self.nonzeroy[good_inds]
    # append x and y if there are points found gotten by good indices
    if np.sum(cur_x):
        x = np.append(x, cur_x.tolist())
        y = np.append(y, cur_y.tolist())
    # store base if there are more 50 points found, otherwise set Noen to it
    if np.sum(cur_x) > 50:
        last_base = np.int(np.mean(cur_x))
    else:
        last_base = None
```

#### Searching Windows

![viz_big_searching_window.png](output_images/viz/viz_big_searching_window.png)

Besides, I added the function `remove_outlier` to remove the outlier, the points out of `95%` of total points, because I took the histogram of each slide. So that there is the chance that the `base`, center point of the searching window, locates at noisy if there is no lane in this slide. And the figure hightlight the points used to fit the polynomial functions.

![viz_lanehightlight.png](output_images/viz/viz_big_lanehightlight.png)

--

### Fit Polynomial

**<sub>line.py: update</sub>**

Fitting polynomial functions are carried out by `polyfit` and `poly1d` **Numpy**. We use `polyfit` to fit 2nd order polynomial with the nonzero points found in the `histogram_detection`. The output of `polyfit` is the coefficients of a 2nd order polynomial. For convenience, I used `poly1d` to create the object which take y values as input to calculate x values. The result of fitting polynomial is shown in the figure below. And I also hightlight the points used to fit the polynomial functions.

![viz_big_fitting_result](output_images/viz/viz_big_fitting_result.png)

--

### Process Improvement

Unlike testing the models with the images, the video output a series of continuous images. I modified `lanefinder` and built `line` model to make final result better.

I added `polynomial_detection` to `lanefinder` to reduce time consuming during the process of lane finding. This function will be carried out if there are lane lines found in previous image. The main idea, different from `histogram_detection`, is to use fitted polynomial function to calculate x value as new `base` with the **central point** in y direction of the `searching window`. And similarily, `remove_outlier` was also applied on the results of `polynomial_detection` to prevent the bad influence of noisy.

The main purpose of using `line` is to make the overlay goes smoothly on the road in the video. The idea is to calculate the average x points with the results of current and previous 5 images as the final output of lane finding. As the approach, I stored the coefficients of the fitted polynomial for 5 images. And then to calculate the average coefficients with the current fitted result with 5 previous ones. Finally, to calculate averaged x points as the final output of lanefinding.

One should be noticed is that I added the weight to the coefficients when calculating the average coefficients. This treatment is used to prevent the overlay goes out of the lane lines at the beginning of turning part. The weight is 0.4 for current coefficients and 0.6 for the previous average coefficients.

### Radius of Curvature and Position Offset

**<sub>line.py: curvature, lanefinder.py: process</sub>**


#### Radius of Curvature
The calculation of **the radius of curvature** was built by following [Interactive Mathmetics:  Applications of Differentiation » 8. Radius of Curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). 

```
radius of curvature = (1 + (dx/dy)^2)^1.5 / (d^2x/dy^2) where x is the output of function y
```

and we use 2nd order polynomial, whoes formula is `x = a*y^2 + b*y + c`

Therefore the foumula is rewritten as follows:

```
radius of curvature = (1 + 2*a*y + b)^1.5 / (2*a)
```

The procedure is shwon below: 

1. Define two converions, `meter per pixel in x direction (xm_per_px)` and `meter per pixel in y direction (ym_per_px)`, whcih are approximately `3.7 / 700` for `xm_per_px` and `30. / 720` for `ym_per_px`.
2. Get the coefficients in meter unit by fitting the polynomial with x and y points multipled by their conversion.
3. The radius is calculated nearest to the car which corresponds to y = 720. 

#### Position Offset

This calculation is quite simple that I defined the center is the **centeral point** nearest to the car between left and right lane lines, which is obtained by the fitted polynomial function with the input `y=720`. And the position of the car is the central point of the image, `image.shape[1]//2`. And therefore the offset is calcuated by the formula, `offset = image.shape[1]/2 - poly(720)`.

--

### Drawing Result

**<sub>lanefinder.py: __put_text, __draw_overlay</sub>**

Before drawing the results as overlay on the image, I firstly hightlighted the lanelines and fill the color in the area between the lane lines. And then I inversedly transformed the warped, masked and highlighted image like the image shown below. I also mentioned inversed transform in section [Perspective Transform](#perspective-transform)

![viz_big_inv_transform_color_area](output_images/viz/viz_big_inv_transform_color_area.png)

As the final step, I used `CV2.addWeight` method to draw the inversedly transformed image on the original image and put the information of `radius of curvature` and `position offset` on the original image by using `CV2.putText` method. The iamge of final output of lanefiding is shown below.

![viz_big_final_result](output_images/viz/viz_big_final_result.png)

## Discussion

As the experience of working on challenge video, my model might be limited to the project vidoe only because most of the parameters of each model have been tuned as I was focusing on project video. And this model totally failed on challenge video that the situations of the road are more complicated. Besides, I was also struggling on improving the performance that the overlay would go out of lane lines at some parts of the road, especially for the beginning of turning parts. In order to solve this problem, I have looked for lots of information on the internet. And therefore, I learned that not only preprocess of images but also postprocess of the "raw" results are quite important.

The follows are possible limitation of my models:

* Continuous big turning.
* Changes of the color hues due to solar altitude.
* Different weather conditions like rain or snow.
* Mixed color of the road.
* Rapid changes of lighting.

And the follows are the possible solutions that make this model better:

* Add `Sobel thresholding`.
* Analyze more color model to imporve the results of `Color thresholding`.
* Preprocess the images with the techniques like `dilate`, `equalize`.
* Postprocess the fitted points with the techniques liek `checking the distance between the lane lines`, `checking whether the found points are plausible lines or not`
* Moving the searching window in x direction for continuous big turning.

## Reflection

By following the instruction of the lesseon, I successfully resolved most of difficulties I encounter. However, I spent much more time, over 20 hours per week, than what I thought on this project, especially for challenge video. Unfortunately, I eventually failed to make the overlay goes well on the raod in challenge video. With limited knowledge of computer vision and processing images, two weeks is totally not enough for me to complete the project and the challenge.

Project video of this project was the challenging video in first project, _**Finding Lane Lines**_. I am happy that I got the satisfied result of this project because I was expecting to learn more about the computer vision to make myself can complete the challenging in project one. As the results of this project, I not only completed the challenge of project 1 but I did better than what I could do in project 1.