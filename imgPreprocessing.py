import cv2
import numpy as np 
import math

#preprocess by thresholding for white
#also do sobel mask for other features
def preprocess(frame):
    h,w = frame.shape[:2]
    finalImg = np.zeros(shape=(h,w),dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #getting white parts of img

    equalized = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    finalImg = cv2.bitwise_or(finalImg, thresh)

    #getting yellow parts of img

    # yellow_HSV_th_min = (0,90,90)
    # yellow_HSV_th_max = (50,255,255)
    # HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # out = cv2.inRange(HSV, yellow_HSV_th_min,yellow_HSV_th_max)
    # finalImg = cv2.bitwise_or(finalImg, out)

    #getting sobel mask
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=9)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)
    sobel_mag = cv2.equalizeHist(sobel_mag)
    _, sobel_mag = cv2.threshold(sobel_mag, 200, 255, cv2.THRESH_BINARY)

    finalImg = cv2.bitwise_or(finalImg, sobel_mag)
    return finalImg

#project to birds eye view for line detection
def birdseyeView(img):
    h, w = img.shape[:2]

    src = np.float32([[w, h-60],    # bottom right
                      [0, h-60],    # bottom left
                      [350, 500],   # top left
                      [800, 500]])  # top right
    dst = np.float32([[w, h],       # bottom right
                      [0, h],       # bottom left
                      [0, 0],       # top left
                      [w, 0]])      # top right

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    return warped, M, Minv