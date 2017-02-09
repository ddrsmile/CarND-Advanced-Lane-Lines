#-*- coding: utf-8 -*-

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    OFFSET = 250
    src = np.int32([
            (132, 703),
            (540, 466),
            (760, 466),
            (1200, 703)])
    dst = np.int32([
        (src[0][0] + OFFSET, 720),
        (src[0][0] + OFFSET, 0),
        (src[-1][0] - OFFSET, 0),
        (src[-1][0] - OFFSET, 720)])


    image = cv2.imread('test_images/test1.jpg')
    #pts1 = np.int32([[500, 480],[780, 480],[1200, 700],[100, 700]])
    #pts2 = np.int32([[132, 703],[540, 466],[740, 466],[1147, 703]])
    #pts3 = np.int32([[0, 0], [1280, 0],[1250, 720],[40, 720]])
    
    img = image.copy()
    #BGR
    cv2.polylines(img, [src], True, (0,0,255), 3)
    cv2.polylines(img, [dst], True, (255,0,0), 3)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()
