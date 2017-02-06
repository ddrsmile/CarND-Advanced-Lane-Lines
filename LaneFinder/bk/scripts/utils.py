# -*- coding: utf-8 -*-
#import required module
import cv2
import numpy as np

def get_idx(base, margin, nonzerox, nonzeroy, y_low, y_high):
    return np.where(((base - margin < nonzerox)&(nonzerox < base + margin)&\
                    ((nonzeroy > y_low) & (nonzeroy < y_high))))

def histogram_dection(image, search_area, n_win, margin=25):
    target_img = image[:, search_area[0]:search_area[1]]
    pixels_per_win = np.int(image.shape[0]/n_win)

    nonzerox, nonzeroy = np.nonzero(np.transpose(image))

    ## declare the containers for histogram_dection
    x = []
    y = []
    idxs = []
    for i in range(n_win):
        start = target_img.shape[0] - (i * pixels_per_win)
        end = start - pixels_per_win

        histogram = np.sum(target_img[end:start, :], axis=0)
        base = np.argmax(histogram)

        idx = get_idx(base, margin, nonzerox, nonzeroy, end, start)

        if np.sum(nonzerox[idx]):
            x = np.append(x, nonzerox[idx].tolist())
            y = np.append(y, nonzeroy[idx].tolist())

    return x.astype(np.float32), y.astype(np.float32)

def fit_poly(x, y):
    ploty = np.linspace(0, self.image.shape[0]-1, self.image.shape[0])
    fit_coeff = np.polyfit(self.l_y, self.l_x, 2)
    return fit_coeff

def get_fitted_line(x, y, ploty):
    for i in range(2):
        fit_x += fit_coeff[i]*ploty**(2-i)
    
    return fit_x