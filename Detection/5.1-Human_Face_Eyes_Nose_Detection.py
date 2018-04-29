# Real-time Human Face, Eyes and Noise Detection
import cv2

import numpy as np

import time

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

eyes_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
 
nose_detect = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

if face_detect.empty():
	raise IOError('Unable to haarcascade_frontalface_alt.xml file')

if eyes_detect.empty():
	raise IOError('Unable to load haarcascade_eye.xml file')
		
if nose_detect.empty():
	raise IOError('Unable to load haarcascade_mcs_nose.xml file')
	
capture = cv2.VideoCapture(0)
time.sleep(2)

while True:
	ret, capturing = capture.read()
	
	resize_frame = cv2.resize(capturing, None, fx=1.0, fy=1.0, 
            interpolation=cv2.INTER_AREA)
   
	gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)

    
	face_detection = face_detect.detectMultiScale(gray, 1.3, 5)
	y = 0
	for (x,y,w,h) in face_detection:
		cv2.rectangle(resize_frame, (x,y), (x+w,y+h), (0,0,255), 10)    

	gray_roi = gray[y:y+h, x:x+w]
	color_roi = resize_frame[y:y+h, x:x+w]
       
	eye_detector = eyes_detect.detectMultiScale(gray_roi)
        		
	for (eye_x, eye_y, eye_w, eye_h) in eye_detector:
		cv2.rectangle(color_roi,(eye_x,eye_y),(eye_x + eye_w, eye_y + eye_h),(255,0,0),5)
			           
	nose_detector = nose_detect.detectMultiScale(gray_roi, 1.3, 5)

	for (nose_x, nose_y, nose_w, nose_h) in nose_detector:
		cv2.rectangle(color_roi, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (0,255,0), 5)

	cv2.imshow("Real-time Detection", resize_frame)

	c = cv2.waitKey(1)
	if c == 27:
		break

capture.release()
cv2.destroyAllWindows() 
