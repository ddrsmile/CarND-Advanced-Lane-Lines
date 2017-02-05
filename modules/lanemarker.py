# -*- coding: utf-8 -*-
#import required module
import cv2
import numpy as np

class LaneMarker(object):
    def __init__(self, calibrator=None, ptransformer=None, image=None):
        self.r_x = []
        self.r_y = []
        self.l_x = []
        self.l_y = []
        self.calibrator = calibrator
        self.ptransformer = ptransformer
        self.image = image
    
    def set_image(self, image):
        self.image = image

    def histogram_detection(self, n_win=10):
        margin = 25
        pixels_per_win = np.int(self.image.shape[0]/n_win)
        nonzerox, nonzeroy = np.nonzero(np.transpose(self.image))

        l_indice = []
        r_indice = []
        for i in range(n_win):
            start = self.image.shape[0] - (i * pixels_per_win)
            end = start - pixels_per_win

            histogram = np.sum(self.image[end:start, :], axis=0)
            midpoint = np.int(histogram.shape[0]/2)
            l_base = np.argmax(histogram[:midpoint])
            r_base = np.argmax(histogram[midpoint:]) + midpoint

            l_idx = np.where(((l_base - margin < nonzerox)&(nonzerox < l_base + margin)&((nonzeroy > end) & (nonzeroy < start))))
            r_idx = np.where(((r_base - margin < nonzerox)&(nonzerox < r_base + margin)&((nonzeroy > end) & (nonzeroy < start))))

            if np.sum(nonzerox[l_idx]):
                self.l_x = np.append(self.l_x, nonzerox[l_idx].tolist())
                self.l_y = np.append(self.l_y, nonzeroy[l_idx].tolist())
            
            if np.sum(nonzerox[r_idx]):
                self.r_x = np.append(self.r_x, nonzerox[r_idx].tolist())
                self.r_y = np.append(self.r_y, nonzeroy[r_idx].tolist())


        self.l_x = self.l_x.astype(np.float32)
        self.l_y = self.l_y.astype(np.float32)
        self.r_x = self.r_x.astype(np.float32)
        self.r_y = self.r_y.astype(np.float32)
        
        
    def fit_polynomial(self):
        ploty = np.linspace(0, self.image.shape[0]-1, self.image.shape[0])
        l_fit = np.polyfit(self.l_y, self.l_x, 2)
        l_fix_x = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
        #l_fitx = l_fit[0]*self.l_y**2 + l_fit[1]*self.l_y + l_fit[2]
        #l_x_int = l_fit[0]*720**2 + l_fit[1]*720 + l_fit[2]

        r_fit = np.polyfit(self.r_y, self.r_x, 2)
        r_fit_x = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]
        #r_fitx = r_fit[0]*self.r_y**2 + r_fit[1]*self.r_y + r_fit[2]
        #r_x_int = r_fit[0]*720**2 + r_fit[1]*720 + r_fit[2]

        return l_fix_x, r_fit_x, ploty

        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from calibrator import Calibrator
    from ptransformer import PTransformer
    from masker import Masker
    import matplotlib.image as mpimg

    calibrator = Calibrator('../scripts/calibration.p')
    src = np.float32([[490, 480],[810, 480],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    
    ptransformer = PTransformer(src=src, dst=dst)

    image = cv2.imread('../test_images/test5.jpg')
    undist = calibrator.undistort(image)
    warped = ptransformer.transform(undist)

    masker = Masker()

    channels = [
        cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)[:,:,2],
        cv2.cvtColor(warped, cv2.COLOR_BGR2LUV)[:,:,0],
        cv2.cvtColor(warped, cv2.COLOR_BGR2Lab)[:,:,2]
    ]

    thresholds = [
        (180, 255),
        (225, 255),
        (155, 200)
    ]

    masker.apply_channel_threshold(channels=channels, thresholds=thresholds)

    binaries = masker.get_binaries()

    combined_binary = masker.combine_binary()

    lanemarker = LaneMarker()
    lanemarker.set_image(combined_binary)

    lanemarker.histogram_detection()
    l_fix_x, r_fit_x, ploty = lanemarker.fit_polynomial()

    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([l_fix_x, ploty])))])
    pts_right = np.array([np.transpose(np.vstack([r_fit_x, ploty]))])
    pts = np.hstack((pts_left, pts_right))
    # color is in the format BGR
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0, 255, 0), thickness = 100)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))
    areawarp = ptransformer.inv_transform(color_warp)
    result = cv2.addWeighted(image, 1, areawarp, 0.5, 0)

    f, (ax1, ax2) = plt.subplots(1,2, figsize=(9, 6))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    ax1.set_xlim(0, 1280)
    ax1.set_ylim(0, 720)
    ax1.plot(l_fix_x, ploty, color='green', linewidth=5)
    ax1.plot(r_fit_x, ploty, color='green', linewidth=5)
    ax1.set_title('Fit Polynomial to Lane Lines', fontsize=16)
    ax1.invert_yaxis() # to visualize as we do the images
    ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax2.set_title('Fill Lane Between Polynomials', fontsize=16)
    #if center < 640:
    #    ax2.text(200, 100, 'Vehicle is {:.2f}m left of center'.format(center*3.7/700),
    #             style='italic', color='white', fontsize=10)
    #else:
    #    ax2.text(200, 100, 'Vehicle is {:.2f}m right of center'.format(center*3.7/700),
    #             style='italic', color='white', fontsize=10)
    #ax2.text(200, 175, 'Radius of curvature is {}m'.format(int((left_curverad + right_curverad)/2)),
    #         style='italic', color='white', fontsize=10)
    #plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype('uint8'))
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.plot(l_fix_x, ploty, color='red', linewidth=5)
    #plt.plot(r_fit_x, ploty, color='red', linewidth=5)

    plt.show()

    '''
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey='col', sharex='row', figsize=(10,4))
    f.tight_layout()
    
    ax1.set_title('Original Image', fontsize=16)
    ax1.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    
    ax2.set_title('Warped Image', fontsize=16)
    ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype('uint8'))
    
    ax3.set_title('Combined color thresholds', fontsize=16)
    ax3.imshow(combined_binary, cmap='gray')


    ax4.set_title('S of HLS', fontsize=16)
    ax4.imshow(binaries[0], cmap='gray')

    ax5.set_title('L of LUV', fontsize=16)
    ax5.imshow(binaries[1], cmap='gray')
    
    ax6.set_title('B of Lab', fontsize=16)
    ax6.imshow(binaries[2], cmap='gray')
    '''
    plt.show()