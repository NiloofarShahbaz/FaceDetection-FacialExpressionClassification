import cv2

face_cascade_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade_detector.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imshow('faces', image)
    key = cv2.waitKey(33)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
