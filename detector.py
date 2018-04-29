import cv2,os
import numpy as np
from PIL import Image 
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataSet'

cam = cv2.VideoCapture(0)

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        if(nbr_predicted==3):
             nbr_predicted='Mantasha'
             #cv2.putText(im, str(nbr_predicted), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
             cv2.putText(im, str(nbr_predicted), (410, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
        elif(nbr_predicted==1):
             nbr_predicted='Prakhar'
             cv2.putText(im, str(nbr_predicted), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
        else:
        #     cv2.putText(im, str("face not Match"), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
            
             #cv2.putText(im, str(nbr_predicted), (210, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
            cv2.putText(im, str(nbr_predicted), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
        #cv2.putText(im, str(nbr_predicted), (210, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
        cv2.imshow('real time face recognition',im)
        
    c = cv2.waitKey(1)
    if c == 27:
        break

cam.release()

cv2.destroyAllWindows()

        
        









