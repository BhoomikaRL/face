import numpy as np
import cv2
import matplotlib.pyplot as plt

cap=cv2.VideoCapture(0)
image = cv2.imread('image_jpg',-1)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faceData = []

faceCount = 0 

ret, frame = cap.read()

grayFace = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cap.release()

faces = faceCascade.detectMultiScale(grayFace,1.3,5)
plt.imshow(frame)
#plt.imshow(grayFace,cmap='gray')

#faces = faceCascade.detectMultiScale(grayFace,1.5,5)

#x,y,w,h=faces[0,:]
names = {
        0:"Person 1",
        1:"Person 2",
        2:"Person 3",
        3:"Person 4",
        4:"Person 5"
        }

i=0
for(x,y,w,h) in faces:
    output = cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    name = names[i]
    cv2.putText(image,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
    i+=1

#croppedFace = frame[y:y+h,x:x+h]
#output = cv2.rectangle(frame,(x,y),(x+w,y+w),(0,255,255),2)

plt.imshow(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))

































