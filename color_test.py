# -*- coding: utf-8 -*-
import cv2
import numpy as np

def apply_threshold(ch, t_min = 0, t_max=255):
    bins = np.zeros_like(ch)
    bins[(ch >= t_min) & (ch <= t_max)] = 1
    return bins

def get_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def get_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

def get_hls(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def get_luv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def apply_mask(img, mask):
    return cv2.bitwise_and(img,img,mask=mask)

def hls_mask(img, min_h, max_h, min_l, max_l, min_s, max_s):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return cv2.inRange(hls, np.array([min_h,min_l,min_s]), np.array([max_h,max_l,max_s]))

def enhance_white_yellow(img, min_l=116, min_s=80):
    yello = hls_mask(img, 13, 24, min_l, 207, min_s, 255)
    white = hls_mask(img, 0, 180, 206, 255, 0, 255)
    mask = cv2.bitwise_or(yello, white)
    return apply_mask(img, mask)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = cv2.imread('test_images/test4.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hsv = get_hsv(img)

    yellow = cv2.inRange(hsv, (20, 50, 150), (40, 255, 255))//255

    hls = get_hls(img)

    white = cv2.inRange(hls, (0, 206, 0), (180, 255, 255))//255

    mask_new = cv2.bitwise_or(yellow, white)

    luv = get_luv(img)
    lab = get_lab(img)

    l = luv[:, :, 0]
    b = lab[:, :, 2]

    l_ch = apply_threshold(l, 215, 255)
    b_ch = apply_threshold(b, 145, 200)
    bins = [l_ch, b_ch, yellow, white]
    mask_old = cv2.bitwise_or(*bins)

    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
    f.tight_layout()
    ax1.imshow(yellow, cmap='gray')
    ax1.set_title('yellow', fontsize=16)
    
    ax2.imshow(white, cmap='gray')
    ax2.set_title('white', fontsize=16)

    ax3.imshow(l_ch, cmap='gray')
    ax3.set_title('l ch', fontsize=16)

    ax4.imshow(b_ch, cmap='gray')
    ax4.set_title('b ch', fontsize=16)

    plt.show()
