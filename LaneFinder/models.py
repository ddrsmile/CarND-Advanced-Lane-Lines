# -*- coding: utf-8 -*-
# import required modules
import cv2
import numpy as np
import pickle

"""
cv2.anything() --> use (width, height)
image.anything() --> use (height, width)
numpy.anything() --> use (height, width)
"""

class Calibrator(object):
    def __init__(self, import_file=None):
        self.obj_points = []
        self.img_points = []
        self.mtx = None
        self.dist = None
        self.image_size = (720, 1280, 3)

        if import_file is not None:
            self.load(import_file)

    def get_corners(self, image, nx=9, ny=6):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.findChessboardCorners(gray, (nx, ny), None)

    def calibrate(self, image_paths, nx=9, ny=6, export=True):
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)


        for index, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)

            # confirm the image size used to calibrate is the same as the image size of viedo
            if image.shape[0] != self.image_size[0] or image.shape[1] != self.image_size[1]:
                print(image.shape)
                image = cv2.resize(image, self.image_size[:-1])
            # find the corners
            ret, corners = self.get_corners(image)

            if ret:
                self.obj_points.append(objp)
                self.img_points.append(corners)
        
        # get mat and dist of the results only
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(self.obj_points, self.img_points, self.image_size[:-1], None, None)

        self.export()

    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
    
    def get_points(self):
        return self.obj_points, self.img_points
    
    def get_calibration(self):
        return self.mtx, self.dist

    def load(self, file_name):
        with open(file_name, 'rb') as f:
            in_dict = pickle.load(f)
        
        # assign the values to the self variables
        try:
            self.obj_points = in_dict['obj_points']
            self.img_points = in_dict['img_points']
            self.mtx = in_dict['mtx']
            self.dist = in_dict['dist']
        except KeyError as e:
            print('There is something wrong when loading the file. {}'.format(e))
            print('Please check the files and load it again.')
            
            # reset the self variables
            self.obj_points = []
            self.img_points = []
            self.mtx = None
            self.dist = None

    def export(self, file_name='calibration.p'):
        # build the dict to store the values
        out_dict = {
            'obj_points': self.obj_points,
            'img_points': self.img_points,
            'mtx': self.mtx,
            'dist': self.dist
        }

        with open(file_name, 'wb') as f:
            pickle.dump(out_dict, file=f)

class PTransformer(object):
    def __init__(self, src=None, dst=None):
        self.src = src
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.inv_M = cv2.getPerspectiveTransform(dst, src)
    
    def set_src(self, src):
        self.src = src
    
    def set_dst(self, dst):
        self.dst = dst
    
    def transform(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    def inv_transform(self, image):
        return cv2.warpPerspective(image, self.inv_M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

class Masker(object):
    def __init__(self):
        self.binaries = []
    
    def extract_channel(self, channel, thresh_min=0, thresh_max=255):
        binary = np.zeros_like(channel)
        binary[(channel >= thresh_min) & (channel <= thresh_max)] = 1
        self.binaries.append(binary)
    
    def combine_binary(self, binaries=None):
        binaries = binaries or self.binaries
        combined_binary = np.zeros_like(binaries[0])
        for binary in binaries:
            combined_binary[(combined_binary == 1) | (binary == 1)] = 1
        return combined_binary
    
    def apply_channel_threshold(self, channels, thresholds):
        for ch, thres in zip(channels, thresholds):
            self.extract_channel(ch, thres[0], thres[1])

    def get_binaries(self):
        return self.binaries

class Line(object):
    def __init__(self, n_image=1, x=None, y=None):
        self.found = False
        self.n_image = n_image
        self.x = x
        self.y = y
        self.n_px_image = []
        self.cur_fit_x = []
        self.avg_fit_x = None
        self.cur_coef = None
        self.avg_coef = None
        self.cur_poly = None
        self.avg_poly = None

        def update(self, x, y):
            self.x = x
            self.y = y

            self.n_px_frame.append(len(self.x))
            self.cur_fit_x.extend(self.x)

            if len(self.n_px_image) > self.n_image:
                posit_to_remove = self.n_px_image.pop(0)
                self.cur_fit_x = self.cur_fit_x[posit_to_remove:]
            
            self.avg_fit_x = np.mean(self.cur_fit_x)
            self.cur_coef = np.polyfit(self.x, self.y, 2)

            if self.avg_coef is None:
                self.avg_coef = self.cur_coef
            else:
                self.avg_coef = (self.avg_coef*(self.n_frame - 1) + self.cur_coef) / self.n_frame
            
            self.cur_poly = np.poly1d(self.cur_coef)
            self.avg_poly = np.poly1d(self.avg_coef)

class LaneFinder(object):
    def __init__(self, calibrator=None, ptransformer=None, masker=None, scan_image_steps=10):
        self.calibrator = calibrator
        self.ptransformer = ptransformer
        self.masker = masker
        self.image = image
        self.nonzerox = None
        self.nonzeroy = None
        self.left = None
        self.right = None
        self.curvature = 0.0
        self.offset = 0.0
    
    def __get_idx(self, base, margin, y_low, y_high):
        return np.where(((base - margin < self.nonzerox)&(self.nonzerox < base + margin)&\
                        ((self.nonzeroy > y_low) & (self.nonzeroy < y_high))))
    
    def __set_nonzero(self, image):
        self.nonzerox, self.nonzeroy = np.nonzero(np.transpose(image))
    
    def histogram_detection(self, image, search_area, steps, margin=25):
        target_img = image[:, search_area[0]:search_area[1]]
        pixels_per_win = np.int(image.shape[0]/steps)

        ## declare the containers for histogram_dection
        x = []
        y = []
        idxs = []
        for i in range(steps):
            start = target_img.shape[0] - (i * pixels_per_win)
            end = start - pixels_per_win

            histogram = np.sum(target_img[end:start, :], axis=0)
            base = np.argmax(histogram)

            idx = self.__get_idx(base, margin, end, start)

            if np.sum(self.nonzerox[idx]):
                x = np.append(x, nonzerox[idx].tolist())
                y = np.append(y, nonzeroy[idx].tolist())

        return x.astype(np.float32), y.astype(np.float32)
    
    def polynomial_detection(self, image, poly, steps, margin=25):
        px_per_step = np.int(image.shape[0]/steps)
        x = []
        y = []

        for i in range(steps):
            start = image.shape[0] - (i * px_per_step)
            end = start - px_per_step

            y_mean = np.mean(start, end)
            x_calc = poly(y_mean)

            idx = self.__get_idx(x_calc, margin, end, start)
            if np.sum(self.nonzerox[idx]):
                x = np.append(x, nonzerox[idx].tolist())
                y = np.append(y, nonzeroy[idx].tolist())

        return x.astype(np.float32), y.astype(np.float32)

    def process(self, image):

        orig_image = np.copy(image)

        if self.calibrator is not None:
            image = self.calibrator.undistort(image)
        
        image = self.ptransformer.transform(image)
        image = self.masker.combine_binary(image)

        self.__set_nonzero(image)

        found_l, found_r = False
        l_x = l_y = r_x = r_y = []

        if self.left is not None and self.right is not None:
            l_x, l_y = self.polynomial_detection(image, self.left.avg_poly, scan_image_steps)
            r_x, r_y = self.polynomial_detection(image, self.right.avg_poly, scan_image_steps)

            found_l = np.sum(l_x) != 0
            found_r = np.sum(r_x) != 0
        
        if not found_l:
            l_x, l_y = self.histogram_detection(image, (0, np.int(image.shape[1]/2)), scan_image_steps)
            found_l = np.sum(l_x) != 0
        
        if not found_r:
            r_x, r_y = self.histogram_detection(image, (np.int(image.shape[1]/2), image.shape[1]), scan_image_steps)
            found_r = np.sum(r_x) != 0
        
        if found_l:
            if self.left:
                self.left.update(l_x, l_y)
            else:
                self.left = Line(self.n_iamge, l_x, l_y)
        
        if found_r:
            if self.right:
                self.right.update(r_x, r_y)
            else:
                self.right = Line(self.n_image, r_x, r_y)

        #TODO put results on the image
        
        return orig_image

