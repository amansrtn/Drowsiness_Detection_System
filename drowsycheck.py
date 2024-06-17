#dlib for deep learning based modules and face landmark detection
import dlib
#for acessing webcam
import cv2
import numpy as np
#face utils for basic conversion
from imutils import face_utils

import run_dataset
# getting the speed of vehcile
speed = int(input("What is the car speed:"))
# starting webcam
cap=cv2.VideoCapture(0)
# now detecting face
detector=dlib.get_frontal_face_detector()
# predicting different landmarks
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# mentioning different stages 
Sleep=0
Drowsy=0
Active=0
status=""
color=(0,0,0)
# calculating the distance betwn two points using eculidian method
def compute(ptA,ptB):
    dist=np.linalg.norm(ptA - ptB)
    return dist
# calculating the eculidian distance of upper and side of eye
def eyegap(a,b,c,d,e,f):
    up=compute(b,d)+compute(c,e)
    down=compute(a,f)
    gap=up/(2.0*down)
    # these ratio are expermintally proven
    if(gap>0.24):
        # eyes open not sleeping
        return 2
    elif(gap>0.20 and gap<=0.24):
        # eyes drowsy may be going to sleep
        return 1
    else:
        # sleeping
        return 0
# main coding
count=0
while True:
    _,frame=cap.read()
    # converting into gray scale for better detection of the face
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    # faces will contain a rectangle where face will be detetcted
    for face in faces:
        # these will calculate the edges of the rectangle
        # x1=face.left()
        # y1=face.top()
        # x2=face.right()
        # y2=face.bottom()
        # now getting landmarks
        landmarks=predictor(gray,face)
        # for landmarks of faces coverting into array
        landmarks=face_utils.shape_to_np(landmarks)
        # now we are intrested in land marks of eye
        # for left eye a=36 b=37 c=38 d=40 e=41 f=39
        left_eye_gap= eyegap(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],landmarks[39])
        # for right eye a=42 b=43 c=44 d=47 e=46 f=45
        right_eye_gap=eyegap(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],landmarks[45])
        # above left and right will get value 1 or 2 or 0 for drowsy active and sleep
        if(left_eye_gap==0 or right_eye_gap==0):
            Sleep+=1
            Active=0
            Drowsy=0
            if(Sleep>10):
                status="SLEEPING!!!"
                count+=1
                color=(255,0,0)
        elif(left_eye_gap==1 or right_eye_gap==1):
            Sleep=0
            Active=0
            count=0
            Drowsy+=1
            if(Drowsy>8):
                status="DROWSY!!"
                color=(0,0,255)
        else:
            Sleep=0
            Active+=1
            count=0
            Drowsy=0
            if(Active>6):
                status="ACTIVE!"
                color=(0,255,0)
        # showing the status of face in the webcam video
        cv2.putText(frame,status,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)
        # for showing the face landmarks on video which is not that necessary below code
        # for i in range(0,68):
        #     (x,y)=landmarks[i]
        #     cv2.circle(frame,(x,y),1,(255,255,255),-1)

        #showing the frame video webcam
        cv2.imshow("FRAME",frame)
        # for stopping the frame
        cv2.waitKey(1) 
        if(0xFF == ord("q")):
            break
    if(count==40):
        break
cap.release()
cv2.destroyAllWindows()
run_dataset.train(speed)






