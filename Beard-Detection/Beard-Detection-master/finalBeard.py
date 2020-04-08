import numpy as np
import cv2

from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model('beard detection.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = frame_gray[y:y+h, x:x+w]    
        roi_gray = cv2.resize(roi_gray,(64,64))
        roi_beard = roi_gray[35:90,7:55]
        roi_beard = cv2.resize(roi_beard,(28,28))
        roi_beard_array = img_to_array(roi_beard)
        roi_beard_array = roi_beard_array/255
        roi_beard_array = np.expand_dims(roi_beard_array,0)
        prediction = model.predict(roi_beard_array)
        if prediction[0][0]<0.5:
            answer = 'Beard'
        else:
            answer = 'Non Beard'
        cv2.putText(frame,answer,(5,70), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),4,cv2.LINE_AA)           
        
       
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()