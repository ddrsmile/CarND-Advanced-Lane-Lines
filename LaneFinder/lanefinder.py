# -*- coding: utf-8 -*-
# import required modules
import cv2
import numpy as np
from LaneFinder.line import Line

class LaneFinder(object):
    def __init__(self, calibrator=None, ptransformer=None, masker=None, n_image=1, scan_image_steps=10, margin=25):
        self.calibrator = calibrator
        self.ptransformer = ptransformer
        self.masker = masker
        self.scan_image_steps = scan_image_steps
        self.margin = margin
        self.nonzerox = None
        self.nonzeroy = None
        self.left = Line(n_image)
        self.right = Line(n_image)
        self.curvature = 0.0
        self.offset = 0.0

    def __get_good_inds(self, base, margin, y_low, y_high):
        return np.where((((base - margin) <= self.nonzerox)&(self.nonzerox <= (base + margin))&\
                        ((self.nonzeroy >= y_low) & (self.nonzeroy <= y_high))))

    def __set_nonzero(self, image):
        self.nonzerox, self.nonzeroy = np.nonzero(np.transpose(image))

    def __color_warp(self, image):
        image_zero = np.zeros_like(image).astype(np.uint8)
        # create a variable with 3 dimension for drawing it on the original image
        color_area = np.dstack((image_zero, image_zero, image_zero))
        # inverse the order of the points to make left and right points become a circle for CV2 operation.
        pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left.avg_fit_x, self.left.y])))])
        pts_right = np.array([np.transpose(np.vstack([self.right.avg_fit_x, self.right.y]))])
        pts = np.hstack((pts_left, pts_right))
        # color is in the format RGB
        cv2.polylines(color_area, np.int_([pts]), isClosed=False, color=(40, 40, 250), thickness = 50)
        cv2.fillPoly(color_area, np.int_([pts]), (250, 40, 40))
        return color_area

    def __put_text(self, image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Radius of Curvature: {:6.2f}m'.format(self.curvature), (700, 50), 
                    font, 1, (255, 255, 255), 2)

        left_or_right = 'left' if self.offset < 0 else 'right'
        cv2.putText(image, 'Position is {0:3.2f}m {1} of center'.format(np.abs(self.offset), left_or_right), 
                   (700, 100), font, 1, (255, 255, 255), 2)

    def __draw_overlay(self, warp, image):
        color_warp = self.__color_warp(warp)
        color_overlay = self.ptransformer.inv_transform(color_warp)
        return cv2.addWeighted(image, 1, color_overlay, 1, 0)

    def histogram_detection(self, image, search_area, steps, margin=25):
        # setup targeted image for searching lane
        target_img = image[:, search_area[0]:search_area[1]]
        # get number of pixels per step in y direction.
        px_per_step = np.int(image.shape[0]/steps)
        # create the containers for storing found points
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)
        # variable for storing the current base
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

        return x.astype(np.float32), y.astype(np.float32)

    def polynomial_detection(self, image, poly, steps, margin=25):
        # get number of pixels per step in y direction.
        px_per_step = np.int(image.shape[0]/steps)
        # create the containers for storing found points
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)

        for i in range(steps):
            # define the range in y direction for searching
            end = image.shape[0] - (i * px_per_step)
            start = end - px_per_step
            # using center point of y direction to find the points in searching area
            y_mean = np.mean([start, end])
            # get x position by fitted polynomial function
            base = poly(y_mean)
            # get the indices in the searching area based on "base" and "margin"
            good_inds = self.__get_good_inds(base, margin, start, end)

            # append x and y if there are points found gotten by good indices
            if np.sum(self.nonzerox[good_inds]):
                x = np.append(x, self.nonzerox[good_inds].tolist())
                y = np.append(y, self.nonzeroy[good_inds].tolist())

        return x.astype(np.float32), y.astype(np.float32)

    def remove_outlier(self, x, y, q=0.5):

        if len(x) == 0 or len(y) == 0:
            return x, y

        # define the range of outliers by the given percentage
        lower_bound = np.percentile(x, q)
        upper_bound = np.percentile(x, 100 - q)

        # remove the outlier
        selection = (x >= lower_bound) & (x <= upper_bound)
        return x[selection], y[selection]

    def process(self, image):
        # back up original image for the output image.
        orig_image = np.copy(image)
        # preprocess the image
        ## undistort
        image = self.calibrator.undistort(image)
        ## perspective transform
        image = self.ptransformer.transform(image)
        ## apply threshold mask to extract lanes
        image = self.masker.get_masked_image(image=image)

        # set nonzero information to find the lanes
        self.__set_nonzero(image)
        # create the containers for storing the coordinates of lanes
        l_x = l_y = r_x = r_y = []
        # get offset caused by perspective transform
        # set the offest 0.5 of original image offest to prevent cut off the lane liens
        img_offset = np.int(self.ptransformer.dst[0][0]*0.5)

        # using fitted polynomial function to find lane if lane is found in the previous image.
        if self.left.found:
            l_x, l_y = self.polynomial_detection(image, self.left.avg_poly, 
                                                 self.scan_image_steps, self.margin)
            self.remove_outlier(l_x, l_y)
            self.left.found = np.sum(l_x) != 0
        
        # using histogram to find lane if there is no lane found or it is the first image of the video.
        if not self.left.found:
            l_x, l_y = self.histogram_detection(image, 
                                                (img_offset, np.int(image.shape[1]/2)), 
                                                self.scan_image_steps, self.margin)
            l_x, l_y = self.remove_outlier(l_x, l_y)
            self.left.found = np.sum(l_x) != 0
        
        # set the previous x and y to the current x and y if there is no point found
        if np.sum(l_y) <= 0:
            l_x = self.left.x
            l_y = self.left.y
        
        # update the information of left lane's coordinates
        self.left.update(l_x, l_y)

        # using fitted polynomial function to find right lane if lane is found in the previous image.
        if self.right.found:
            r_x, r_y = self.polynomial_detection(image, self.right.avg_poly, 
                                                 self.scan_image_steps, self.margin)
            r_x, r_y = self.remove_outlier(r_x, r_y)
            self.right.found = np.sum(r_x) != 0
        
        # using histogram to find lane if there is no lane found or it is the first image of the video.
        if not self.right.found:
            r_x, r_y = self.histogram_detection(image, 
                                                (np.int(image.shape[1]/2), image.shape[1] - img_offset),
                                                self.scan_image_steps, self.margin)
            self.remove_outlier(r_x, r_y)
            self.right.found = np.sum(r_x) != 0
        
        # set the previous x and y to the current x and y if there is no point found
        if np.sum(r_y) <= 0:
            r_x = self.right.x
            r_y = self.right.y
        
        # update the information of right lane's coordinates
        self.right.update(r_x, r_y)
        
        # obtain the radius of curvature and the position offset.
        ## set calculated radius to self.curvature
        self.curvature = np.mean([self.left.curvature, self.right.curvature])
        center_poly = (self.left.avg_poly + self.right.avg_poly) /2
        ## set calculated offset to self.offset
        self.offset = (image.shape[1] / 2 - center_poly(720)) * 3.7 / 700

        # draw overlay on the image
        orig_image = self.__draw_overlay(image, orig_image)
        # put radius of curvature and position offset on the image
        self.__put_text(orig_image)
        return orig_image
