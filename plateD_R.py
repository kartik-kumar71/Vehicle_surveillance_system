import cv2
import numpy as np
from numpy.core.fromnumeric import ndim
from numpy.matrixlib.defmatrix import _convert_from_string
import pytesseract

# cap = cv2.VideoCapture('video.mov')
cap = cv2.VideoCapture('PATH_TO_VIDEO')

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

plates_cascade = cv2.CascadeClassifier('CASCADE CLASSIFIER PATH')
# plates_cascade = cv2.CascadeClassifier('K:/python/Lib/site-packages/cv2/data/haarcascade_russian_plate_number.xml')
# plates_cascade = cv2.CascadeClassifier('K:/python/Lib/site-packages/cv2/data/haarcascade_licence_plate_rus_16stages.xml')

while(cap.isOpened()):
    ret, frame = cap.read()
    img = cv2.resize(frame, (500,400), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original Video', img)
    plates = plates_cascade.detectMultiScale(gray, 1.2, 4)
    for (x,y,w,h) in plates:
        plates_rec = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        
        gray_plates = gray[y:y+h, x:x+w]
        color_plates = img[y:y+h, x:x+w]
        
        height, width = gray_plates.shape
        print(height, width)
        cv2.imshow('Detected plate', plates_rec)
        # cv2.waitKey(0)
    # print('Number of detected licence plates:', len(plates))

    # extraction
    for x,y,w,h in plates:
        plate_image = gray[y+3:y+h, x+12:x+w-1]
        en_w = int(plate_image.shape[1]*2)
        en_h = int(plate_image.shape[0]*2)
        dim = (en_w, en_h)
        plate_image = cv2.resize(plate_image, dim, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('number plate', plate_image)
        #img to text
        result = pytesseract.image_to_string(plate_image, lang='eng')
        print('Number plate text : ',result)
        
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()