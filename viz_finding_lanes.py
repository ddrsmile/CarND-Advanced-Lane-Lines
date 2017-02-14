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
    [595, 450],
    [690, 450],
    [1115, 720],
    [216, 720]
])

dst = np.float32([
    [450, 0],
    [830, 0],
    [830, 720],
    [450, 720]
])

# create required objects to visualize fitting polynomial
calibrator = Calibrator('calibration.p')
ptransformer = PTransformer(src=src, dst=dst)
masker = Masker()

# define the basic parameter for searching lane
scan_image_steps=10
margin=50
nonzerox = None
nonzeroy = None

def get_binary_warped(image):
    image = calibrator.undistort(image)
    save_single_image(image, 'undistorted')
    warp = ptransformer.transform(image)
    save_single_image(warp, 'perspective_transform')
    binary_warped = masker.get_masked_image(warp)
    return binary_warped

def set_nonzeros(image):
    nonzerox, nonzeroy = np.nonzero(np.transpose(image))
    return nonzerox, nonzeroy

def get_good_inds(base, margin, y_low, y_high):
    return np.where((((base - margin) <= nonzerox)&(nonzerox <= (base + margin))&\
                    ((nonzeroy >= y_low) & (nonzeroy <= y_high))))

def save_single_image(image, fname, fitted_lines=None):
    fig = plt.figure(figsize=(8, 4.5))
    plt.imshow(image)
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    if fitted_lines is not None:
        ## draw fitted lines
        plt.plot(fitted_lines['l_x'], fitted_lines['ploty'], color='yellow')
        plt.plot(fitted_lines['r_x'], fitted_lines['ploty'], color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    plt.savefig('output_images/viz/viz_big_{}.png'.format(fname), 
                bbox_inches='tight',
                transparent="True", 
                pad_inches=0)
    
    

def histogram_detection(viz_img, image, search_area, steps, margin=25):
    # setup targeted image for searching lane
    target_img = image[:, search_area[0]:search_area[1]]
    # get number of pixels per step in y direction.
    px_per_step = np.int(image.shape[0]/steps)
    # create the containers for storing found points
    x = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)

    last_base = None

    for i in range(steps):
        # define the range in y direction for searching
        end = target_img.shape[0] - (i * px_per_step)
        start = end - px_per_step
        # set last_base to current base if there are more 50 points found in previous image
        if last_base is None:
            histogram = np.sum(target_img[start:end, :], axis=0)
            # add search_area[0], image offset in x direction, 
            # to ensure the positions of points are correct.
            base = np.argmax(histogram) + search_area[0]
        else:
            base = last_base
        # draw searching window
        cv2.rectangle(viz_img, (base-margin,start),(base+margin, end),(255,125,0), 2)
        # get the indices in the searching area based on "base" and "margin"
        good_inds = get_good_inds(base, margin, start, end)
        cur_x, cur_y = nonzerox[good_inds], nonzeroy[good_inds]

        # append x and y if there are points found gotten by good indices
        if np.sum(cur_x):
            x = np.append(x, cur_x.tolist())
            y = np.append(y, cur_y.tolist())
        
        if np.sum(cur_x) > 50:
            last_base = np.int(np.mean(cur_x))
        else:
            last_base = None

    return x.astype(np.float32), y.astype(np.float32)

def remove_outlier(x, y, q=0.5):

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
    input_image = image.copy()
    save_single_image(input_image, 'input_image')

    # get masked and warped image in 3 dimensional form
    binary_warped = get_binary_warped(image)
    # create fit_img to visualize the fitting process
    fit_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    save_single_image(fit_img, 'binary_warped')
    

    # draw the target area where lanefinder searches the lane
    img_offset = np.int(dst[0][0]*0.5)
    targeted = fit_img.copy()
    cv2.rectangle(targeted,(img_offset,0),(image.shape[1]//2 - 5, image.shape[0]),(255,0,0), 5)
    cv2.rectangle(targeted,(image.shape[1]//2 + 5,0),(image.shape[1] - img_offset, image.shape[0]),(0,0,255), 5)
    save_single_image(targeted, 'search_target')

    # set nonzeros for searching lanes
    nonzerox, nonzeroy = set_nonzeros(binary_warped)

    # create the containers to store current found points
    l_x = l_y = r_x = r_y = []

    # draw searching area along the y direction
    ## left
    l_x, l_y = histogram_detection(targeted, binary_warped, 
                                   (img_offset, image.shape[1]//2), 
                                   steps=scan_image_steps, margin=margin)
    ## remove outlier
    l_x, l_y = remove_outlier(l_x, l_y)
    
    ## right 
    r_x, r_y = histogram_detection(targeted, binary_warped, 
                                   (image.shape[1]//2, image.shape[1] - img_offset), 
                                   steps=scan_image_steps, margin=margin)
    ## remove outlier
    r_x, r_y = remove_outlier(r_x, r_y)
    
    save_single_image(targeted, 'searching_window')

    ## draw the found points
    targeted[l_y.astype(np.int32), l_x.astype(np.int32)] = [255, 0, 0]
    targeted[r_y.astype(np.int32), r_x.astype(np.int32)] = [0, 0, 255]

    save_single_image(targeted, 'lanehightlight')

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

    ## create dict for drawing fitted lines on the image
    fitted_lines = {
        'l_x': l_fit_x,
        'r_x': r_fit_x,
        'ploty': ploty
    }

    ## draw final result
    ## hightlight found points
    fit_img[l_y.astype(np.int32), l_x.astype(np.int32)] = [255, 0, 0]
    fit_img[r_y.astype(np.int32), r_x.astype(np.int32)] = [0, 0, 255]

    ## draw fitted area
    window_img = np.zeros_like(fit_img)
    left_line_window1 = np.array([np.transpose(np.vstack([l_fit_x-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([l_fit_x+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([r_fit_x-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([r_fit_x+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    fitting_result = cv2.addWeighted(fit_img, 1, window_img, 0.3, 0)

    save_single_image(fitting_result, 'fitting_result', fitted_lines)

    # draw fill area crossed by fitted lines and inv-transforming
    image_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_area = np.dstack((image_zero, image_zero, image_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([l_fit_x, ploty])))])
    pts_right = np.array([np.transpose(np.vstack([r_fit_x, ploty]))])
    pts = np.hstack((pts_left, pts_right))
    cv2.polylines(color_area, np.int_([pts]), isClosed=False, color=(40, 40, 250), thickness = 50)
    cv2.fillPoly(color_area, np.int_([pts]), (250, 40, 40))
    color_area = ptransformer.inv_transform(color_area)
    save_single_image(color_area, 'inv_transform_color_area')


    # calculate radius of curvature and position offset
    ym_per_px = 30. / 720. # meters per pixel in y dimension
    xm_per_px = 3.7 / 700. # meters per pixel in x dimension

    y = np.array(np.linspace(0, 720, num=100))
    y_eval = np.max(y)

    # calculate radius of curvature of left lane
    l_x = np.array(list(map(left_poly, y)))
    l_cur_coef = np.polyfit(y * ym_per_px, l_x * xm_per_px, 2)
    l_curverad = ((1 + (2 * l_cur_coef[0] * y_eval / 2. + l_cur_coef[1]) ** 2) ** 1.5) / np.absolute(2 * l_cur_coef[0])
    # calculate radius of curvature of right lane
    r_x = np.array(list(map(right_poly, y)))
    r_cur_coef = np.polyfit(y * ym_per_px, r_x * xm_per_px, 2)
    r_curverad = ((1 + (2 * r_cur_coef[0] * y_eval + r_cur_coef[1]) ** 2) ** 1.5) / np.absolute(2 * r_cur_coef[0])
    # get average radius of curvature
    curverad = np.mean([l_curverad, r_curverad])

    center_poly = (left_poly + right_poly) /2
    ## set calculated offset to self.offset
    pos_offset = (input_image.shape[1] / 2 - center_poly(719)) * xm_per_px

    # put overlay on original image
    final_result = cv2.addWeighted(input_image, 1, color_area, 1, 0)

    # put calculated radius of curvature and position offset on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_result, 'Radius of Curvature: {:6.2f}m'.format(curverad), (700, 50), 
                font, 1, (255, 255, 255), 2)

    left_or_right = 'left' if pos_offset < 0 else 'right'
    cv2.putText(final_result, 'Position is {0:3.2f}m {1} of center'.format(np.abs(pos_offset), left_or_right), 
                (700, 100), font, 1, (255, 255, 255), 2)
    save_single_image(final_result, 'final_result')
    
    # show all the figures
    plt.show()
    # clear the figure
    plt.clf()