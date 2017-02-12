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
        # ensure image_paths is list type
        # if it is not list then make it as list
        if not isinstance(image_paths, list):
            image_paths = list(image_paths)
        for index, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)

            # confirm the image size used to calibrate is the same as the image size of viedo
            if image.shape[0] != self.image_size[0] or image.shape[1] != self.image_size[1]:
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