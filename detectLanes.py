import cv2
import numpy as np 
import math
from collections import deque
import sys
np.set_printoptions(threshold=sys.maxsize)

class Line():
    def __init__(self):
        self.detected = False
        self.coefficients = None
        self.recentCoefs = deque(maxlen=10)
        self.xCoords = None
        self.yCoords = None

    def update(self, newCoefs, detected):
        self.recentCoefs.append(newCoefs)
        self.detected = detected

    def avgCoefs(self):
        #avg over columns of recentCoefs, i.e. over each coef
        return np.mean(self.recentCoefs, axis = 0)
        

#input: birdseye view
def slidingWindowSearch(img, leftLaneLine, rightLaneLine, numWindows=9):
    #first calculate x midpoint of left lane and right lane
    #get list of nonzero pixels in img, split into x and y
    #identify height of window in prop to height of img and num windows
    #start for loop through num windows, and define borders of this window width 200 (margin 100)
    #find coords that are nonzero inside this window
    #if there are > 50px on a specific line, then it has been detected as part of line
        #in this case, set new x midpoint as the mean of these x nonzeros
        #append these indices to some overall array of the pixels in the lane
    height, width = img.shape[:2]

    #sum over columns in bottom half of img
    density = np.sum(img[height//2:-30, :], axis=0)
    midpoint = width//2

    #identify x coordinates of start of left and right lanes
    leftLaneBaseX = np.argmax(density[:midpoint])
    rightLaneBaseX = np.argmax(density[midpoint:]) + midpoint
    currLeftLaneX = leftLaneBaseX
    currRightLaneX = rightLaneBaseX

    nonzeroPts = img.nonzero()
    nonzeroX = np.array(nonzeroPts[0])
    nonzeroY = np.array(nonzeroPts[1])

    windowHeight = height // numWindows
    #how wide 
    margin = 100
    #need at least 50px detected to call it a line
    detectionThreshold = 20

    leftLaneIdx = []
    rightLaneIdx = []

    res = np.dstack((img, img, img)) * 255

    for window in range(numWindows):
        #defining current window boundaries
        yHigh = height - (window) * windowHeight
        yLow = height - (window + 1) * windowHeight

        leftLaneXLeft = currLeftLaneX - margin
        leftLaneXRight = currLeftLaneX + margin

        rightLaneXLeft = currRightLaneX - margin
        rightLaneXRight = currRightLaneX + margin

        cv2.rectangle(res, (leftLaneXLeft, yLow), (leftLaneXRight, yHigh), (0, 255, 0), 2)
        cv2.rectangle(res, (rightLaneXLeft, yLow), (rightLaneXRight, yHigh), (0, 255, 0), 2)

        leftLaneGood = ((nonzeroX >= leftLaneXLeft) & (nonzeroX < leftLaneXRight) &
                       (nonzeroY >= yLow) & (nonzeroY < yHigh)).nonzero()[0]
        rightLaneGood = ((nonzeroX >= rightLaneXLeft) & (nonzeroX < rightLaneXRight) & 
                        (nonzeroY >= yLow) * (nonzeroY < yHigh)).nonzero()[0]

        # print(window, yLow, yHigh, leftLaneXLeft, leftLaneXRight, np.count_nonzero((nonzeroY < yHigh) & (nonzeroX < leftLaneXRight)))
        leftLaneIdx.append(leftLaneGood)
        rightLaneIdx.append(rightLaneGood)

        if len(leftLaneGood) > detectionThreshold:
            currLeftLaneX = np.int(np.mean(nonzeroX[leftLaneGood]))
        if len(rightLaneGood) > detectionThreshold:
            currRightLaneX = np.int(np.mean(nonzeroX[rightLaneGood]))
    
    leftLaneIdx = np.concatenate(leftLaneIdx)
    rightLaneIdx = np.concatenate(rightLaneIdx)

    leftLaneLine.xCoords, leftLaneLine.yCoords = nonzeroX[leftLaneIdx], nonzeroY[leftLaneIdx]
    rightLaneLine.xCoords, rightLaneLine.yCoords = nonzeroX[rightLaneIdx], nonzeroY[rightLaneIdx]

    detected = True
    if not list(leftLaneLine.xCoords) or not list(leftLaneLine.yCoords):
        left_coefficients = leftLaneLine.coefficients
        detected = False
    else:
        left_coefficients = np.polyfit(leftLaneLine.xCoords, leftLaneLine.yCoords, 2)
    
    if not list(rightLaneLine.xCoords) or not list(rightLaneLine.yCoords):
        right_coefficients = rightLaneLine.coefficients
        detected = False
    else:
        right_coefficients = np.polyfit(rightLaneLine.xCoords, rightLaneLine.yCoords, 2)
    
    leftLaneLine.update(left_coefficients, detected)
    rightLaneLine.update(right_coefficients, detected)

    res[nonzeroY[leftLaneIdx], nonzeroX[leftLaneIdx]] = [0,255,0]
    res[nonzeroY[rightLaneIdx], nonzeroX[rightLaneIdx]] = [0,0,255]
    # res[300,nonzeroX[(nonzeroY == 300)]] = [255,0,0]
    # print(img[300][50:250])
    # print(nonzeroX[(nonzeroY == 300)])

    return res, leftLaneLine, rightLaneLine