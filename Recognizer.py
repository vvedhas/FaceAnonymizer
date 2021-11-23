import cv2
import numpy as np
from cv2.data import haarcascades

faceCascade = cv2.CascadeClassifier(haarcascades+'./haarcascade_frontalface_default.xml')


video = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
pw,ph=(8,8)
while True:
    ret, im =video.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        temp = cv2.resize(im[y:y+h,x:x+w], (pw, ph), interpolation=cv2.INTER_LINEAR)
        im[y:y+h,x:x+w]=cv2.resize(temp,(w,h),interpolation=cv2.INTER_NEAREST)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==27:
        break
video.release()
cv2.destroyAllWindows()
