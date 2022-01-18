import os
import cv2 as cv
import numpy as np

people = ['Angry', 'Fear',
          'Happy', 'Neutral', 'Sad', 'Suprise']
DIR = './Training'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

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
            # use haar cascade to get faces and use them as the training set
            faces_roi = gray[y:y+h, x:x+w]
            features.append(faces_roi)
            labels.append(label)

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, labels)
print('Training done ---------------')
face_recognizer.save('face_trained.yml')