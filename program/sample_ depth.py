import numpy as np
import cv2
import matplotlib.pyplot as plt

imgL = cv2.imread('imgfile/imgl.png', 0 )
imgR = cv2.imread('imgfile/imgr.png', 0 )

stereo = cv2.StereoBM_create(numDisparities= 16, blockSize = 7)
disparity = stereo.compute(imgL, imgR)

plt.xticks([]), plt.yticks([])
plt.imshow(disparity,'gray')
plt.show()
