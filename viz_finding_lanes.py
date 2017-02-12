# -*- coding: utf-8 -*-
# import required modules
import cv2
import numpy as np
import pickle

from LaneFinder.calibrator import Calibrator
from LaneFinder.ptransformer import PTransformer
from LaneFinder.masker import Masker
from LaneFinder.lanefinder import LaneFinder
from LaneFinder.line import Line

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

# create required objects to visualize fitting polynomial
calibrator = Calibrator('calibration.p')
ptransformer = PTransformer(src=src, dst=dst)
masker = Masker()

# define the basic parameter for searching lane
scan_image_steps=10
margin=25
nonzerox = None
nonzeroy = None

def get_binary_warped(image):
    image = calibrator.undistort(image)
    warp = ptransformer.transform(image)
    binary_warped = masker.get_masked_image(warp)
    return binary_warped

def set_nonzeros(image):
    nonzerox, nonzeroy = np.nonzero(np.transpose(image))
    return nonzerox, nonzeroy

def get_good_inds(base, margin, y_low, y_high):
    return np.where((((base - margin) <= nonzerox)&(nonzerox <= (base + margin))&\
                    ((nonzeroy >= y_low) & (nonzeroy <= y_high))))

def histogram_detection(viz_img, image, search_area, steps, margin=25):
    # setup targeted image for searching lane
    target_img = image[:, search_area[0]:search_area[1]]
    # get number of pixels per step in y direction.
    px_per_step = np.int(image.shape[0]/steps)
    # create the containers for storing found points
    x = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)

    for i in range(steps):
        # define the range in y direction for searching
        end = target_img.shape[0] - (i * px_per_step)
        start = end - px_per_step

        histogram = np.sum(target_img[start:end, :], axis=0)
        # add search_area[0], image offset in x direction, 
        # to ensure the positions of points are correct.
        base = np.argmax(histogram) + search_area[0]
        # draw searching window
        cv2.rectangle(viz_img, (base-margin,start),(base+margin, end),(255,125,0), 2)
        # get the indices in the searching area based on "base" and "margin"
        good_inds = get_good_inds(base, margin, start, end)

        # append x and y if there are points found gotten by good indices
        if np.sum(nonzerox[good_inds]):
            x = np.append(x, nonzerox[good_inds].tolist())
            y = np.append(y, nonzeroy[good_inds].tolist())

    return x.astype(np.float32), y.astype(np.float32)

def remove_outlier(x, y, q=5):

    if len(x) == 0 or len(y) == 0:
        return x, y

    # define the range of outliers by the given percentage
    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)

    # remove the outlier
    selection = (x >= lower_bound) & (x <= upper_bound)
    return x[selection], y[selection]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = cv2.imread('test_images/test4.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_image = image.copy()

    # get masked and warped image in 3 dimensional form
    binary_warped = get_binary_warped(image)
    # create out_img to visualization
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #plt.imshow(out_img)
    #plt.show()
    
    

    # draw the target area where lanefinder searches the lane
    img_offset = np.int(dst[0][0]*0.3)
    targeted = out_img.copy()
    cv2.rectangle(targeted,(img_offset,0),(image.shape[1]//2 - 5, image.shape[0]),(255,0,0), 5)
    cv2.rectangle(targeted,(image.shape[1]//2 + 5,0),(image.shape[1] - img_offset, image.shape[0]),(0,0,255), 5)
    #plt.imshow(targeted)
    #plt.show()

    # set nonzeros for searching lanes
    nonzerox, nonzeroy = set_nonzeros(binary_warped)
    # create line objects for both left and right lane
    left = Line()
    right = Line()

    # create the containers to store current found points
    l_x = l_y = r_x = r_y = []

    # draw searching area along the y direction
    ## left
    l_x, l_y = histogram_detection(targeted, binary_warped, 
                                   (img_offset, image.shape[1]//2), 
                                   steps=scan_image_steps, margin=25)
    ## remove outlier
    l_x, l_y = remove_outlier(l_x, l_y)
    ## draw the found points
    targeted[l_y.astype(np.int32), l_x.astype(np.int32)] = [255, 0, 0]
    ## right 
    r_x, r_y = histogram_detection(targeted, binary_warped, 
                                   (image.shape[1]//2, image.shape[1] - img_offset), 
                                   steps=scan_image_steps, margin=25)
    ## remove outlier
    r_x, r_y = remove_outlier(r_x, r_y)
    ## draw the found points
    targeted[r_y.astype(np.int32), r_x.astype(np.int32)] = [0, 0, 255]

    #plt.imshow(targeted)
    #plt.show()

    # fit polynomial
    ## left
    left_coef = np.polyfit(l_y, l_x, 2)
    left_poly = np.poly1d(left_coef)

    ## right 
    right_coef = np.polyfit(r_y, r_x, 2)
    right_poly = np.poly1d(right_coef)

    ## get fitted points
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    l_fit_x = left_poly(ploty)
    r_fit_x = right_poly(ploty)


    ## draw final result
    ## hightlight found points
    out_img[l_y.astype(np.int32), l_x.astype(np.int32)] = [255, 0, 0]
    out_img[r_y.astype(np.int32), r_x.astype(np.int32)] = [0, 0, 255]

    ## draw fitted area
    window_img = np.zeros_like(out_img)
    left_line_window1 = np.array([np.transpose(np.vstack([l_fit_x-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([l_fit_x+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([r_fit_x-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([r_fit_x+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)

    ## draw fitted lines
    plt.plot(l_fit_x, ploty, color='yellow')
    plt.plot(r_fit_x, ploty, color='yellow')
    
    ## set up axis
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


