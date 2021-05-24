import cv2
import numpy as np 
import math

cap = cv2.VideoCapture('labeled/3.hevc')
# dataFile = open("labeled/3.txt")
lines = []
with open("labeled/3.txt") as f:
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
    reg2, thresh = cv2.threshold(equalized, 250, 255, cv2.THRESH_BINARY)
    # ret2, thresh = cv2.threshold(gray, 130, 145, cv2.THRESH_BINARY)
    finalImg = cv2.bitwise_or(finalImg, thresh)

    #getting yellow parts of img
    yellow_HSV_th_min = (0,70,70)
    yellow_HSV_th_max = (50,255,255)
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = cv2.inRange(HSV, yellow_HSV_th_min,yellow_HSV_th_max)
    finalImg = cv2.bitwise_or(finalImg, out)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(finalImg.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    finalImg = cv2.bitwise_or(finalImg, closing)
    
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