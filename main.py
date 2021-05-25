import cv2
import numpy as np 
import math

fileToUse = 8
folderToUse = "labeled/" if fileToUse <= 5 else "unlabeled/"
cap = cv2.VideoCapture(folderToUse + str(fileToUse) + ".hevc")
# dataFile = open("labeled/3.txt")
lines = []
if fileToUse <= 5:
    with open(folderToUse + str(fileToUse) + ".txt") as f:
        lines = [line.rstrip().split() for line in f]

i=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h,w = frame.shape[:2]
    finalImg = np.zeros(shape=(h,w),dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #getting white parts of img

    equalized = cv2.equalizeHist(gray)
    ret2, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    finalImg = cv2.bitwise_or(finalImg, thresh2)

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
    
    if fileToUse <= 5:
        yawRads = float(lines[i][1])
        cv2.putText(finalImg, 
                    "radians: " + str(round(yawRads,4)), 
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,255,255), 2, cv2.LINE_4)
        cv2.putText(finalImg,
                    "degrees: " + str(round(yawRads*180/math.pi, 4)), 
                    (50,80), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,255,255), 2, cv2.LINE_4)

    cv2.imshow('frame', finalImg)
    # cv2.moveWindow('frame', -1300,-600)
    cv2.moveWindow('frame', 100, -100)

    #quit on 'q' press
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    i += 1

cap.release()
cv2.destroyAllWindows()