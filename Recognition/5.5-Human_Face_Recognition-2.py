
# Real-time Human Face Recognition - 2
# Training using face images stored in human_faces folder

import cv2

import numpy as np

# From Operating System(os) to return a list containing names
# of the entries in the directory given by path - os.listdir(path)
from os import listdir
 
# os.path.isfile(path) - Returns True if path is an existing file
from os.path import isfile, join

# Face images for training are taken from human_faces folder
path = './human_faces/'

# To filter only files in the specified path we use:
path_files = [f for f in listdir(path) if isfile(join(path, f))]

Training, Index = [], []


for i, files in enumerate(path_files):
    path_image = path + path_files[i]
    
    # Train images are read from path_image and converted to gray
    train_images = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
    
    # Convert train images into numpy array using np.asarray and
    # append it with Training array 
    # Training.append(np.array(train_images, dtype)
    # dtype=unit8 is an unsigned 8 bit integer (0 to 255)
    Training.append(np.asarray(train_images, dtype=np.uint8))
    
    Index.append(i)

# Numpy array is created for Index using np.asarray
# np.array(Index, dtype)
# dtype=np.int32 is an 32 bit integer
Index = np.asarray(Index, dtype=np.int32)


# Histogram - Graphical representation of tonal distribution in image

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the face_recognizer 
face_recognizer.train(np.asarray(Training), np.asarray(Index))
print("Training completed successfully")

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if face_detect.empty():
	raise IOError('Unable to haarcascade_frontalface_default.xml file')

def face_detector(image, size=0.5):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
        
    face_detection = face_detect.detectMultiScale(gray, 1.3, 5)

    if face_detection is ():
        return image, []
    
    for (x,y,w,h) in face_detection:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),4)
        
        cropped = image[y:y+h, x:x+w]
        
        cropped = cv2.resize(cropped, (250, 250))
    
    return image, cropped

capture = cv2.VideoCapture(0)
while True:
	ret, capturing = capture.read()
    
	image, faces = face_detector(capturing)
    
	try:
		faces = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)

        # Faces is passed to the prediction model
		matching = face_recognizer.predict(faces)
        # matching tuple contains the index and the score (confidence) value 
        
		if matching[1] < 500:
			score = int( 100 * (1 - (matching[1])/350) )
			string = str(score) + '% Matching Confidence'
        
		if score > 70:
			cv2.putText(image, string, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
			cv2.putText(image, "Welcome Mantasha", (210, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
			
			cv2.imshow('Real-time Face Recognition', image)
		
		else:
			cv2.putText(image, "This is NOT Mantasha", (150, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
			cv2.imshow('Real-time Face Recognition', image)

	except:
		cv2.putText(image, "FACE NOT FOUND ", (150, 250) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
		cv2.imshow('Real-time Face Recognition', image)
		pass
        
	c = cv2.waitKey(1)
	if c == 27:
		break
        
capture.release()

cv2.destroyAllWindows()     
