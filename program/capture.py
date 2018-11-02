#capture 
import numpy as np
import cv2

left = cv2.VideoCapture(0)
right = cv2.VideoCapture(2)

LEFT_PATH = "C:/Users/user/1.open_cv/capture/left/{:06d}.jpg"
RIGHT_PATH = "C:/Users/user/1.open_cv/capture/right/{:06d}.jpg"

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

frameId = 0

while(True):
    if not (left.grab() and right.grab()):
        print("No more frames")
        break

    _, leftFrame = left.retrieve()
    leftFrame = cropHorizontal(leftFrame)
    _, rightFrame = right.retrieve()
    rightFrame = cropHorizontal(rightFrame)

    cv2.imwrite(LEFT_PATH.format(frameId), leftFrame)
    cv2.imwrite(RIGHT_PATH.format(frameId), rightFrame)

    cv2.imshow('left', leftFrame)
    cv2.imshow('right', rightFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frameId += 1

left.release()
right.release()
cv2.destroyAllWindows()