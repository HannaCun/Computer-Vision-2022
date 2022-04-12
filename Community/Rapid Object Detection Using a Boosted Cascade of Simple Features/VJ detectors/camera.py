import cv2 as cv


def face_detect_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2BGRA)
    face_detector = cv.CascadeClassifier("C:/Users/GG/Downloads/dhfhub-opencv-master/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml")
    faces = face_detector.detectMultiScale(gray,1.02,5)
    for x,y,w,h in faces:
        cv.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    cv.imshow("result",image)

# src = cv.imread("001test.jpg")
# cv.namedWindow("input_image",cv.WINDOW_AUTOSIZE)
# cv.imshow("input_image",src)
capture = cv.VideoCapture(0)
cv.namedWindow("result",cv.WINDOW_AUTOSIZE)

while (True):
    ret,frame=capture.read()
    frame=cv.flip(frame,1)
    face_detect_demo(frame)
    if cv.waitKey(1) == ord('q'):
        break
cv.waitKey(0)

cv.destroyAllWindows()