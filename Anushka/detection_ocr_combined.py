#Required Installations

! pip install easyocr
! sudo apt install tesseract-ocr

#imports

import imutils
from PIL import Image
import numpy as np
import argparse
import imutils
import easyocr
import matplotlib.pyplot as plt
import cv2 as cv
import os

#Reading the image and preprocessing

img=cv.imread('PATH_TO_IMAGE')
#img=img[:,:200]
#plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
#print(img.shape)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#gray=gray[int(img.shape[0]/2):]
plt_img=cv.cvtColor(gray,cv.COLOR_BGR2RGB)
#plt.imshow(plt_img)
blur=cv.bilateralFilter(gray,30,34,25)
blur=cv.GaussianBlur(blur,(3,3),cv.BORDER_DEFAULT)
blur=cv.bilateralFilter(blur,10,10,25)
edged=cv.Canny(blur,30,255)
#plt.imshow(cv.cvtColor(edged,cv.COLOR_BGR2RGB))

#Detecting and grabing the effective contours 

keypoints=cv.findContours(edged.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
contours=imutils.grab_contours(keypoints)

# Sorting according to the area and detection of NumberPlate coordinates

contours=sorted(contours,key=cv.contourArea,reverse=True)[:10]
#print(contours)
location=None
for cons in contours:
  #approximate=cv.approxPolyDP(cons, 0.1* cv.arcLength(cons, True), True)
  approx=cv.approxPolyDP(cons,10,True)
  print(len(approx))
  if len(approx)==4:
    location=approx
    break
print(location)

#Masking of the approximate numberplate with blank image

mask=np.zeros(gray.shape,dtype='uint8')
img2=cv.drawContours(mask,[location],0,255,-1)
img2=cv.bitwise_and(gray,gray,mask=mask)
#plt.imshow(cv.cvtColor(img2,cv.COLOR_BGR2RGB))
(x,y)=np.where(mask==255)
(x1,y1)=(np.min(x),np.min(y))
(x2,y2)=(np.max(x),np.max(y))
plate=gray[x1:x2+1,y1:y2+1]
#plt.imshow(cv.cvtColor(plate,cv.COLOR_BGR2RGB))
#plt.imshow(cv.cvtColor(plate,cv.COLOR_BGR2RGB))

#Recognizing characters using easy-ocr

re=easyocr.Reader(['en'])
result=re.readtext(plate)
res=''
for i in range(0,len(result)):
  print(result[i][1])
  res=res+result[i][1]
#res=res.rstrip()
font=cv.FONT_HERSHEY_PLAIN
y=img.shape[1]
x=img.shape[0]

#Labeling of image with number-plate characters 
final=cv.putText(img,text=res,org=(int(y1+(0.002)*y),int(x1-(0.006)*x)),fontFace=font,fontScale=int(0.01*x),color=[0,255,0],thickness=2)
final=cv.rectangle(img,tuple(approx[0][0]),tuple(approx[2][0]),[0,255,0],2)
plt.imshow(cv.cvtColor(final,cv.COLOR_BGR2RGB))
