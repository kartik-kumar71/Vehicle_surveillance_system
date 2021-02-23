import cv2

cam = cv2.VideoCapture('../test.mp4')

while True:
	ret, frame = cam.read()
	cv2.imshow('image',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cam.release()
cv2.destroyAllWindows()