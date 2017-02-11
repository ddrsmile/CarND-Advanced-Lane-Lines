# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np

src = np.float32([
    [590, 450],
    [720, 450],
    [1115, 720],
    [170, 720],
    
])

if __name__ == "__main__":
    import matplotlib.pyplot as plt


    img_path = sys.argv[1]
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.polylines(image, np.int_([src]), isClosed=True, color=(40, 40, 250), thickness = 10)

    plt.imshow(image)
    plt.show()