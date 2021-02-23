from anpr import ANPR
from imutils import paths
import imutils
import cv2

cam = cv2.VideoCapture('test.mp4')
anpr = ANPR()
def cleanup_text(text):
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()



while True:
	ret, image = cam.read()
	image = imutils.resize(image, width=600)
	(lpText, lpCnt) = anpr.find_and_ocr(image)
	
	if lpText is not None and lpCnt is not None:
		
		box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
		box = box.astype("int")
		cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
		
		(x, y, w, h) = cv2.boundingRect(lpCnt)
		cv2.putText(image, cleanup_text(lpText), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
		print(lpText)
		cv2.imshow("image", image)
	cv2.imshow('image',image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cam.release()
cv2.destroyAllWindows()