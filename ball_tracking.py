import cv2 as cv
import numpy as np
from collections import deque
import time
cap = cv.VideoCapture(0)


while True:
    ret, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv = cv.GaussianBlur(hsv, (11,11),0)
    lowerGreen = (29,86,6)
    upperGreen = (64,255,255)
    pts = deque(maxlen= 32)       
    mask = cv.inRange(hsv, lowerGreen, upperGreen) 
    c,h = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(c, key =cv.contourArea)
    ((x,y), rad) = cv.minEnclosingCircle(cnt)
    m = cv.moments(cnt)
    center = (int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"]))
    cv.circle(frame, (int(x), int(y)), int(rad), (0,255,0), 2)

    cv.imshow('frame',frame)
    key = cv.waitKey(5)
    if key == 27:
       break    


