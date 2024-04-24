#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


# In[ ]:


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8,maxHands=4)
folder="C:/Users/91983/OneDrive/Desktop/imagecollestion"
of=20
isize=300



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
      
             
        else:
            k=isize/w
            hCal=math.ceil(k * h)
            imgresize = cv2.resize(imgCrop,(isize,hCal))
            imgresizeshape = imgresize.shape
            hGap=math.ceil((isize-hCal)/2)
            imgwhite[hGap:hCal + hGap, :]=imgresize
            
        cv2.imshow("image Crop",imgCrop)
        cv2.imshow(" image white", imgwhite)
      
       
        
    cv2.imshow("Image",img)
    key=cv2.waitKey(1)
    if key==ord("g"):
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',imgCrop)


# In[ ]:





# In[ ]:





# In[ ]:




