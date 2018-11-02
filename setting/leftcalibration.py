import numpy as np
import cv2
import glob
import os

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] 
imgpoints = [] 

a = 0

images = glob.glob('*.jpg')
for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (7,6), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        a = a+1
    else:
        os.remove("C:/Users/user/1.open_cv/capture/left/"+fname)
        
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
np.savez('cleft.npz',ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, encoding = 'UTF-8', fmt='1.0f')

#img = cv2.imread ( '000013.jpg' )
#h, w = img.shape [: 2]
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix (mtx, dist, (w, h), 1, (w, h))
#dst = cv2.undistort (img, mtx, dist, None , newcameramtx)
     
#x, y, w, h = roi
#dst = dst [y : y + h, x : x + w]
#cv2.imwrite ( 'calibresult.png' , dst)

#if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
        
#cv2.destroyAllWindows()