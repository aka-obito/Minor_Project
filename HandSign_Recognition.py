#!/usr/bin/env python
# coding: utf-8

# In[1]:



import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


# In[ ]:


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8,maxHands=2)
classifier = Classifier("C:/Users/91983/OneDrive/Desktop/Model/keras_model.h5","C:/Users/91983/OneDrive/Desktop/Model/labels.txt")
of=20
isize=300
counter=0
labels=[ "A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","X","Y","W","Z"]

while True:
    success, img =cap.read()
    imgoutput=img.copy()   
    hands, img = detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h =hand['bbox']
        
        imgwhite=np.ones((isize,isize,3),np.uint8)*255
        imgCrop =img [ y -of:y +h + of, x-of:x + w +of]
        
        imgCropshape = imgCrop.shape
        
        aspectratio = h/w;
        if aspectratio>1:
            k=isize/h
            wCal=math.ceil(k * w)
            imgresize = cv2.resize(imgCrop,(wCal, isize))
            imgresizeshape = imgresize.shape
            wGap=math.ceil((isize-wCal)/2)
            imgwhite[:, wGap:wCal+wGap]=imgresize
            prediction , index = classifier.getPrediction(imgwhite)
            print(prediction,index)
            
             
        else:
            k=isize/w
            hCal=math.ceil(k * h)
            imgresize = cv2.resize(imgCrop,(isize,hCal))
            imgresizeshape = imgresize.shape
            hGap=math.ceil((isize-hCal)/2)
            imgwhite[hGap:hCal + hGap, :]=imgresize
            prediction , index = classifier.getPrediction(imgwhite)
            
          
        cv2.putText(imgoutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgoutput,(x-of,y-of),(x+w+of,y+h+of),(255,0,255),4)
       
        
    cv2.imshow("Image",imgoutput)
    cv2.waitKey(1)

    def main():
        st.title("sfgs")
        html_body="""<body style="background-color:red;"> </body> """
        st.markdown(html_body)


# In[ ]:





# In[ ]:





# In[ ]:




