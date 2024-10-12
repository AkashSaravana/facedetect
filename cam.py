import cv2  #module

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #Trained model

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert into gray

    faces = face_cascade.detectMultiScale(gray, 1.1, 3)  # match

    for (x, y, w, h) in faces:  # loop
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),3)

    cv2.imshow('live face detection', img)
    if cv2.waitKey(20) ==ord('x'):
        break

cap.release()
cv2.destroyAllWindows()



