import os
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Angry', 'Fear',
          'Happy', 'Neutral', 'Sad', 'Suprise']
DIR = './Testing'

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

i = 0  # total number of images (more precisely the number of detected faces)
j = 0  # correctly classfied images

for person in people:
    path = os.path.join(DIR, person)
    label = people.index(person)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        img_array = cv.imread(img_path)

        gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

        faces_rect = haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces_rect:
            i += 1
            faces_roi = gray[y:y+h, x:x+w]
            prediction, confidence = face_recognizer.predict(faces_roi)
            if prediction == label:
                j += 1

print(f'i = {i}')
print(f'j = {j}')
print(f'the final prediction accuracy is {j / i}.')
