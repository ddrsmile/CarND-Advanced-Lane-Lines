# -*- coding: utf-8 -*-
# import required modules
import numpy as np

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

        # initialize line object if there are x and y.
        if x is not None:
            self.update(x, y)

    @property
    def avg_poly(self):
        return np.poly1d(self.avg_coef)

    @property
    def cur_poly(self):
        try:
            return np.poly1d(self.coef[-1])
        except IndexError:
            return None

    @property
    def curvature(self):
        # define conversions in x and y from pixels space to meters
        ym_per_px = 30. / 720. # meters per pixel in y dimension
        xm_per_px = 3.7 / 700. # meters per pixel in x dimension

        # get latest fitted polynomial function
        cur_poly = self.cur_poly

        # return 0 if there is no coefficient of fitted polynomial
        if cur_poly is None:
            return 0.
        # cover the same range of images
        y = np.array(np.linspace(0, 720, num=100))
        x = np.array(list(map(cur_poly, y)))
        y_eval = np.max(y)
        cur_poly = np.polyfit(y * ym_per_px, x * xm_per_px, 2)
        curverad = ((1 + (2 * cur_poly[0] * y_eval + cur_poly[1]) ** 2) ** 1.5) / np.absolute(2 * cur_poly[0])
        return curverad

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

        # update self.avg_coef if there is previous one and n_image is greater 1
        if self.avg_coef is not None and self.n_image > 1:
            weight = 0.4
            self.avg_coef = (self.avg_coef * weight + cur_coef * (1 - weight))
        else:
            self.avg_coef = cur_coef

        avg_poly = np.poly1d(self.avg_coef)
        self.avg_fit_x = avg_poly(self.y)

        if len(self.coef) > self.n_image:
            self.coef.pop(0)

        if len(self.top) > self.n_image:
            self.top.pop(0)
            self.bottom.pop(0)