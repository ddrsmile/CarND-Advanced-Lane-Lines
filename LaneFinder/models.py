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
    def __init__(self, color_models=[], nth_elements=[], thresholds=[]):
        self.color_models = color_models
        self.nth_elements = nth_elements
        self.thresholds = thresholds
    
    def apply_threshold(self, channel, thresh_min=0, thresh_max=255):
        binary = np.zeros_like(channel)
        binary[(channel >= thresh_min) & (channel <= thresh_max)] = 1
        return binary

    def set_channels(self, warpped_image, color_models=None, nth_elements=None):
        channels = []
        # use self.color_models if color_models is not given
        color_models = color_models or self.color_models
        # use self.nth_elements if nth_elements is not given
        nth_elements = nth_elements or self.nth_elements
        for model, element in zip(color_models, nth_elements):
            channels.append(cv2.cvtColor(warpped_image, model)[:,:,element])

        return channels
    
    def build_binary_with_thresholds(self, channels=None, thresholds=None):
        thresholds = thresholds or self.thresholds
        binaries = []
        for channel, threshold in zip(channels, thresholds):
            binaries.append(self.apply_threshold(channel, threshold[0], threshold[1]))
        return binaries
    
    def combine_binaries(self, binaries=None):
        combined_binary = np.zeros_like(binaries[0])
        for binary in binaries:
            combined_binary[(combined_binary == 1) | (binary == 1)] = 1
        return combined_binary

    def get_masked_image(self, image):
        channels = self.set_channels(image)
        binaries = self.build_binary_with_thresholds(channels=channels)
        combined_binary = self.combine_binaries(binaries=binaries)
        return combined_binary

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

        if x is not None:
            self.update(x, y)

    def update(self, x, y):
        self.x = x
        self.y = y

        self.n_px_image.append(len(self.x))
        self.cur_fit_x.extend(self.x)

        if len(self.n_px_image) > self.n_image:
            posit_to_remove = self.n_px_image.pop(0)
            self.cur_fit_x = self.cur_fit_x[posit_to_remove:]
        
        self.avg_fit_x = np.mean(self.cur_fit_x)
        self.cur_coef = np.polyfit(self.y, self.x, 2)

        if self.avg_coef is None:
            self.avg_coef = self.cur_coef
        else:
            self.avg_coef = (self.avg_coef*(self.n_image - 1) + self.cur_coef) / self.n_image
        
        self.cur_poly = np.poly1d(self.cur_coef)
        self.avg_poly = np.poly1d(self.avg_coef)

class LaneFinder(object):
    def __init__(self, calibrator=None, ptransformer=None, masker=None, n_image=1, scan_image_steps=10):
        self.calibrator = calibrator
        self.ptransformer = ptransformer
        self.masker = masker
        self.scan_image_steps = scan_image_steps
        self.n_image = n_image
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
    
    def __get_mask(self, image):
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
        mask_zero = np.zeros_like(image).astype(np.uint8)
        color_mask = np.dstack((mask_zero, mask_zero, mask_zero))
        pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left.avg_poly(ploty), ploty])))])
        pts_right = np.array([np.transpose(np.vstack([self.right.avg_poly(ploty), ploty]))])
        pts = np.hstack((pts_left, pts_right))
        # color is in the format BGR
        cv2.polylines(color_mask, np.int_([pts]), isClosed=False, color=(0, 255, 0), thickness = 100)
        cv2.fillPoly(color_mask, np.int_([pts]), (0, 0, 255))
        return color_mask


    def __annotate_image(self, mask, image):
        color_mask = self.__get_mask(mask)
        areawarp = self.ptransformer.inv_transform(color_mask)
        return cv2.addWeighted(image, 1, areawarp, 0.5, 0)

    def histogram_detection(self, image, search_area, steps, margin=25):
        target_img = image[:, search_area[0]:search_area[1]]
        pixels_per_win = np.int(image.shape[0]/steps)

        ## declare the containers for histogram_dection
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)
        idxs = []
        for i in range(steps):
            start = target_img.shape[0] - (i * pixels_per_win)
            end = start - pixels_per_win

            histogram = np.sum(target_img[end:start, :], axis=0)
            base = np.argmax(histogram) + search_area[0]

            idx = self.__get_idx(base, margin, end, start)

            if np.sum(self.nonzerox[idx]):
                x = np.append(x, self.nonzerox[idx].tolist())
                y = np.append(y, self.nonzeroy[idx].tolist())

        return x.astype(np.float32), y.astype(np.float32)
    
    def polynomial_detection(self, image, poly, steps, margin=25):
        px_per_step = np.int(image.shape[0]/steps)
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)

        for i in range(steps):
            start = image.shape[0] - (i * px_per_step)
            end = start - px_per_step
            y_mean = np.mean([start, end])
            x_calc = poly(y_mean)

            idx = self.__get_idx(x_calc, margin, end, start)
            if np.sum(self.nonzerox[idx]):
                x = np.append(x, self.nonzerox[idx].tolist())
                y = np.append(y, self.nonzeroy[idx].tolist())

        return x.astype(np.float32), y.astype(np.float32)

    def process(self, image):

        orig_image = np.copy(image)

        if self.calibrator is not None:
            image = self.calibrator.undistort(image)
        
        image = self.ptransformer.transform(image)
        image = self.masker.get_masked_image(image)
        self.__set_nonzero(image)

        found_l = found_r = False
        l_x = l_y = r_x = r_y = []

        if self.left is not None and self.right is not None:
            l_x, l_y = self.polynomial_detection(image, self.left.avg_poly, self.scan_image_steps)
            r_x, r_y = self.polynomial_detection(image, self.right.avg_poly, self.scan_image_steps)

            found_l = np.sum(l_x) != 0
            found_r = np.sum(r_x) != 0
        
        if not found_l:
            l_x, l_y = self.histogram_detection(image, (0, np.int(image.shape[1]/2)), self.scan_image_steps)
            found_l = np.sum(l_x) != 0
        
        if not found_r:
            r_x, r_y = self.histogram_detection(image, (np.int(image.shape[1]/2), image.shape[1]), self.scan_image_steps)
            found_r = np.sum(r_x) != 0
        
        if found_l:
            if self.left:
                self.left.update(l_x, l_y)
            else:
                self.left = Line(self.n_image, l_x, l_y)
        
        if found_r:
            if self.right:
                self.right.update(r_x, r_y)
            else:
                self.right = Line(self.n_image, r_x, r_y)

        #TODO put results on the image
        if self.left is not None and self.right is not None:
            orig_image = self.__annotate_image(image, orig_image)

        return orig_image
