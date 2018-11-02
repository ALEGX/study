##camera set
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

##find algorithm
def canny(img): 
    img = cv2.Canny(img, 50, 200)
    return img

def corner(img):
    img2 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    imgray = np.float32(imgray)
    dst= cv2.cornerHarris(imgray, 2, 3, 0.04)
    dst = cv2.dilate(dst,None)
    
    img2[dst > 0.01*dst.max()] = [0,0,255]
    
    return img2

def Fast_True(img):
    imgray = cv2.FastFeatureDetector_create(30)
    img2 = None
    fast = cv2.FastFeatureDetector_create(30)
    
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, img2, (255,0,0))
    return img2

def Fast_False(img):
    imgray = cv2.FastFeatureDetector_create(30)
    img2 = None
    fast = cv2.FastFeatureDetector_create(30)
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, img2, (255,0,0))
    return img2

def clahe(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
    img2 = clahe.apply(imgray)  
    res=np.hstack((imgray, img2)) 
    return res

        
left_save = np.load('C:/Users/user/Desktop/nps/cleft.npz')
right_save = np.load('C:/Users/user/Desktop/nps/cleft.npz')

left = cv2.VideoCapture(2)
right = cv2.VideoCapture(0)

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

CROP_WIDTH = 1920
def cropHorizontal(image):
    return image[:,
            int((CAMERA_WIDTH-CROP_WIDTH)/2):
            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]
while(True):
    if not (left.grab() and right.grab()):
        print("No more frames")
        break
    
    _, leftFrame = left.retrieve()
    leftFrame = cropHorizontal(leftFrame)
    _, rightFrame = right.retrieve()
    rightFrame = cropHorizontal(rightFrame)
    
    
    h1, w1 = leftFrame.shape [: 2]
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix (left_save['mtx'], left_save['dist'], (w1, h1), 1, (w1, h1))

    mapx,mapy = cv2.initUndistortRectifyMap(left_save['mtx'],left_save['dist'],None,newcameramtx,(w1,h1),5)     
    img1 = cv2.remap(leftFrame,mapx,mapy,cv2.INTER_LINEAR)  

    x1,y1,w1,h1 = roi    
    img1 = img1[y1:y1+h1, x1:x1+w1]
    
    ##img1(left) set
    #img1 = canny(img1)
    #img1 = Fast_False(img1)
    #img1 = Fast_True(img1)
    #img1 = corner(img1)
    #img1 = clahe(img1)
    ##
                
    h2, w2 = rightFrame.shape [: 2]
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix (right_save['mtx'], right_save['dist'], (w2, h2), 1, (w2, h2))

    mapx,mapy = cv2.initUndistortRectifyMap(right_save['mtx'],right_save['dist'],None,newcameramtx,(w2,h2),5)     
    img2 = cv2.remap(rightFrame,mapx,mapy,cv2.INTER_LINEAR)  

    x2,y2,w2,h2 = roi    
    img2 = img2[y2:y2+h2, x2:x2+w2]
                
    ##img1(left) set
    #img2 = clahe(img2)
    img2 = canny(img2)
    #img2 = Fast_False(img2)
    #img2 = Fast_True(img2)
    #img2 = corner(img2)
    ##

    cv2.imshow('left', img1)
    cv2.imshow('right', img2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left.release()
right.release()
cv2.destroyAllWindows()