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
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    yawRads = float(lines[i][1])
    cv2.putText(frame, 
                "radians: " + str(round(yawRads,4)), 
                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255,255), 2, cv2.LINE_4)
    cv2.putText(frame,
                "degrees: " + str(round(yawRads*180/math.pi, 4)), 
                (50,80), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255,255), 2, cv2.LINE_4)

    cv2.imshow('frame', frame)
    cv2.moveWindow('frame', -1300,-600)

    #quit on 'q' press
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
    i += 1

cap.release()
cv2.destroyAllWindows()