from keras.models import load_model
import cv2
import numpy as np

model = load_model('model-009.model')

#cascade Classifer
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while(True):
    # get each frame from camera
    ret,img=source.read()
    # convert it to grey
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detecct faces
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
        # crop the face only
        face_img=gray[y:y+w,x:x+w]
        #resize it
        resized=cv2.resize(face_img,(100,100))
        # normalized
        normalized=resized/255.0
        # convert it to 4D array
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()