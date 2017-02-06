# -*- coding: utf-8 -*-

class Line(object):
    def __init__(self, n_frame=1, x=None, y=None):
        self.n_frame = n_frame
        self.x = x
        self.y = y
        self.cur_fit = None
        self.avg_fit = None
        self.cur_poly = None
        self.avg_poly = None