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
    def get_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def get_lab(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    def get_hls(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    def get_luv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

    def apply_threshold(self, channel, thresh_min=0, thresh_max=255):
        binary = np.zeros_like(channel)
        binary[(channel >= thresh_min) & (channel <= thresh_max)] = 1
        return binary

    def set_channels(self, warpped_image, color_models=None, nth_chs=None):
        channels = []
        # use self.color_models if color_models is not given
        color_models = color_models or self.color_models
        # use self.nth_elements if nth_elements is not given
        nth_chs = nth_chs or self.nth_chs
        for model, nth_ch in zip(color_models, nth_chs):
            channels.append(cv2.cvtColor(warpped_image, model)[:,:,nth_ch])

        return channels
    
    def build_binary_with_thresholds(self, channels=None, thresholds=None):
        thresholds = thresholds or self.thresholds
        binaries = []
        for channel, threshold in zip(channels, thresholds):
            binaries.append(self.apply_threshold(channel, threshold[0], threshold[1]))
        return binaries
    
    def combine_binaries(self, binaries=None):
        #combined_binary = np.zeros_like(binaries[0])
        #for binary in binaries:
        #    combined_binary[(combined_binary == 1) | (binary == 1)] = 1
        combined_binary = cv2.bitwise_or(*binaries)
        return combined_binary
    
    def get_masked_image(self, image):
        #channels = self.set_channels(image)
        #binaries = self.build_binary_with_thresholds(channels=channels)
        binaries = []
        binaries.append(self.extract_yellow(image))
        binaries.append(self.extract_white(image))
        combined_binary = self.combine_binaries(binaries=binaries)
        return combined_binary

    def extract_yellow(self, image):
        hsv = self.get_hsv(image)
        yellow = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))//255
        return yellow
    
    def extract_white(self, image):
        hls = self.get_hls(image)
        white = cv2.inRange(hls, (0, 206, 0), (180, 255, 255))//255
        return white


class Line(object):
    def __init__(self, n_image=1, x=None, y=None):
        self.found = False
        self.n_image = n_image
        self.x = x
        self.y = y
        self.top = []
        self.bottom = []
        self.coef = []
        self.avg_fit_x = None
        self.avg_coef = None

        if x is not None:
            self.update(x, y)
    
    @property
    def avg_poly(self):
        return np.poly1d(self.avg_coef)
    
    @property
    def get_cur_poly(self):
        try:
            return np.poly1d(self.coef[-1])
        except IndexError:
            return None
        
    
    def update(self, x, y):
        cur_x = x
        cur_y = y
        
        cur_coef = np.polyfit(cur_y, cur_x, 2)
        self.coef.append(cur_coef)
        cur_poly = np.poly1d(cur_coef)
        cur_top = cur_poly(0)
        cur_bottom = cur_poly(719)

        self.top.append(cur_top)
        self.bottom.append(cur_bottom)

        cur_top = np.mean(self.top)
        cur_bottom = np.mean(self.bottom)
        cur_x = np.append(cur_x, [cur_top, cur_bottom])
        cur_y = np.append(cur_y, [0, 719])
        sorted_idx = np.argsort(cur_y)
        self.x = cur_x[sorted_idx]
        self.y = cur_y[sorted_idx]

        if self.avg_coef is None:
            self.avg_coef = cur_coef
        else:
            weight = 0.4
            self.avg_coef = (self.avg_coef * weight + cur_coef * (1 - weight))

        #self.avg_coef = [
        #    np.mean([n for n, _, _ in self.coef]),
        #    np.mean([n for _, n, _ in self.coef]),
        #    np.mean([n for _, _, n in self.coef])
        #]

        avg_poly = np.poly1d(self.avg_coef)
        self.avg_fit_x = avg_poly(self.y)

        if len(self.coef) > self.n_image:
            self.coef.pop(0)

        if len(self.top) > self.n_image:
            self.top.pop(0)
            self.bottom.pop(0)
    
    @property
    def curvature(self):
        # define conversions in x and y from pixels space to meters
        ym_per_px = 30. / 720. # meters per pixel in y dimension
        xm_per_px = 3.7 / 700. # meters per pixel in x dimension

        # get latest fitted polynomial function
        cur_poly = self.get_cur_poly

        # return 0 if there is no coefficient of fitted polynomial
        if cur_poly is None:
            return 0.
        # cover the same range of images
        y = np.array(np.linspace(0, 719, num=100))
        x = np.array(list(map(cur_poly, y)))
        y_eval = np.max(y)
        cur_poly = np.polyfit(y * ym_per_px, x * xm_per_px, 2)
        curverad = ((1 + (2 * cur_poly[0] * y_eval / 2. + cur_poly[1]) ** 2) ** 1.5) / np.absolute(2 * cur_poly[0])
        return curverad


class LaneFinder(object):
    def __init__(self, calibrator=None, ptransformer=None, masker=None, n_image=1, scan_image_steps=10, margin=25):
        self.calibrator = calibrator
        self.ptransformer = ptransformer
        self.masker = masker
        self.scan_image_steps = scan_image_steps
        self.n_image = n_image
        self.margin = margin
        self.nonzerox = None
        self.nonzeroy = None
        self.left = Line(n_image)
        self.right = Line(n_image)
        self.curvature = 0.0
        self.offset = 0.0
    
    def __get_good_inds(self, base, margin, y_low, y_high):
        return np.where((((base - margin) < self.nonzerox)&(self.nonzerox < (base + margin))&\
                        ((self.nonzeroy > y_low) & (self.nonzeroy < y_high))))
    
    def __set_nonzero(self, image):
        self.nonzerox, self.nonzeroy = np.nonzero(np.transpose(image))
    
    def __color_warp(self, image):
        image_zero = np.zeros_like(image).astype(np.uint8)
        color_area = np.dstack((image_zero, image_zero, image_zero))
        pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left.avg_fit_x, self.left.y])))])
        pts_right = np.array([np.transpose(np.vstack([self.right.avg_fit_x, self.right.y]))])
        pts = np.hstack((pts_left, pts_right))
        # color is in the format RGB
        cv2.polylines(color_area, np.int_([pts]), isClosed=False, color=(40, 40, 250), thickness = 50)
        cv2.fillPoly(color_area, np.int_([pts]), (250, 40, 40))
        return color_area

    def __put_text(self, image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Radius of Curvature = %d(m)' % self.curvature, (50, 50), font, 1, (255, 255, 255), 2)
        left_or_right = 'left' if self.offset < 0 else 'right'
        cv2.putText(image, 'Vehicle is %.2fm %s of center' % (np.abs(self.offset), left_or_right), (50, 100), font, 1,
                    (255, 255, 255), 2)
    
    def __draw_overlay(self, warp, image):
        color_warp = self.__color_warp(warp)
        color_overlay = self.ptransformer.inv_transform(color_warp)
        return cv2.addWeighted(image, 1, color_overlay, 1, 0)

    def histogram_detection(self, image, search_area, steps, margin=25):
        target_img = image[:, search_area[0]:search_area[1]]
        pixels_per_win = np.int(image.shape[0]/steps)

        ## declare the containers for histogram_dection
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)

        for i in range(steps):
            end = target_img.shape[0] - (i * pixels_per_win)
            start = end - pixels_per_win

            histogram = np.sum(target_img[start:end, :], axis=0)
            base = np.argmax(histogram) + search_area[0]

            good_inds = self.__get_good_inds(base, margin, start, end)

            if np.sum(self.nonzerox[good_inds]):
                x = np.append(x, self.nonzerox[good_inds].tolist())
                y = np.append(y, self.nonzeroy[good_inds].tolist())

        return x.astype(np.float32), y.astype(np.float32)
    
    def polynomial_detection(self, image, poly, steps, margin=25):
        px_per_step = np.int(image.shape[0]/steps)
        x = np.array([], dtype=np.float32)
        y = np.array([], dtype=np.float32)

        for i in range(steps):
            end = image.shape[0] - (i * px_per_step)
            start = end - px_per_step
            y_mean = np.mean([start, end])
            base = poly(y_mean)

            good_inds = self.__get_good_inds(base, margin, start, end)

            if np.sum(self.nonzerox[good_inds]):
                x = np.append(x, self.nonzerox[good_inds].tolist())
                y = np.append(y, self.nonzeroy[good_inds].tolist())


        return x.astype(np.float32), y.astype(np.float32)
    
    def remove_outlier(self, x, y, q=5):
        if len(x) == 0 or len(y) == 0:
            return x, y
        
        lower_bound = np.percentile(x, q)
        upper_bound = np.percentile(x, 100 - q)
        selection = (x >= lower_bound) & (x <= upper_bound)
        return x[selection], y[selection]


    def process(self, image):

        orig_image = np.copy(image)

        image = self.calibrator.undistort(image)        
        image = self.ptransformer.transform(image)
        image = self.masker.get_masked_image(image)

        self.__set_nonzero(image)

        found_l = found_r = False
        l_x = l_y = r_x = r_y = []
        
        if self.left.found:
            l_x, l_y = self.polynomial_detection(image, self.left.avg_poly, self.scan_image_steps, self.margin * 3)
            self.left.found = np.sum(l_x) != 0
        
        if not self.left.found:
            l_x, l_y = self.histogram_detection(image, (0, np.int(image.shape[1]/2)), self.scan_image_steps, self.margin)
            self.left.found = np.sum(l_x) != 0
            self.remove_outlier(l_x, l_y)
        
        if np.sum(l_y) <= 0:
            l_x = self.left.x
            l_y = self.left.y

        if self.right.found:
            r_x, r_y = self.polynomial_detection(image, self.right.avg_poly, self.scan_image_steps, self.margin * 3)
            self.right.found = np.sum(r_x) != 0

        if not self.right.found:
            r_x, r_y = self.histogram_detection(image, (np.int(image.shape[1]/2), image.shape[1]), self.scan_image_steps, self.margin)
            self.right.found = np.sum(r_x) != 0
            self.remove_outlier(r_x, r_y)
        
        if not np.sum(r_y) > 0:
            r_x = self.right.x
            r_y = self.right.y

        self.left.update(l_x, l_y)
        self.right.update(r_x, r_y)
        
        # obtain the information of curvature and the offset of the car from the center.
        self.curvature = np.mean([self.left.curvature, self.right.curvature])
        center_poly = (self.left.avg_poly + self.right.avg_poly) /2
        self.offset = (image.shape[1] / 2 - center_poly(719)) * 3.7 / 700

        #TODO put info on the image
        orig_image = self.__draw_overlay(image, orig_image)
        self.__put_text(orig_image)
        return orig_image
