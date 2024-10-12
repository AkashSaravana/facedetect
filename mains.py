import cv2  #module

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #Trained model

cap = cv2.VideoCapture('CCTV.mp4')

while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert into gray

    faces = face_cascade.detectMultiScale(gray, 1.09, 3)     # match CCTV 1.09 ,3  times 1.15 ,4

    for (x, y, w, h) in faces:  # loop
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),4)

    cv2.imshow('video capture', img)
    if cv2.waitKey(20) ==ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
