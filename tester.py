import cv2
import os
import numpy as np
import faceRecognition as fr


# This module takes images  stored in disk and performs face recognition
confidence_level = 39  # (100 - confidence_level)% match
test_person = 'Om'
test_img = cv2.imread('TestImages/' + test_person + '.jpg')      # test_img path
faces_detected, gray_img = fr.faceDetection(test_img)
print("faces_detected:", faces_detected)

# creating dictionary containing names for each label
name = {0: "Priyanka", 1: "Kangana", 2: "Dhiraj", 3: "Om", 4: "Tanushree"}
# name = {0: "", 1: "", 2: "Dhiraj", 3: "Om", 4: "Tanushree"}


# # Comment belows lines when running this program second time.
# # Since it saves training.yml file in directory
# faces, faceID = fr.labels_for_training_data('trainingImages', name)
# face_recognizer = fr.train_classifier(faces, faceID)
# # remove existing training yml
# if os.path.exists('trainingData.yml'):
#     os.remove('trainingData.yml')
# face_recognizer.write('trainingData.yml')


# Uncomment below line for subsequent runs
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# use this to load training data for subsequent runs
face_recognizer.read('trainingData.yml')

for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y:y+h, x:x+h]
    # predicting the label of given image
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence:", confidence)
    print("label:", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    print(predicted_name)
    # If confidence more than 37 then don't print predicted face text on screen means 62%
    if confidence < confidence_level:
        fr.put_text(test_img, predicted_name, x, y)
    # when matching is less but guess is correct
    elif confidence >= confidence_level and predicted_name == test_person:
        fr.put_text(test_img, predicted_name, x, y)
    else:
        continue

resized_img = cv2.resize(test_img, (800, 800))
cv2.imshow("face detection tutorial", resized_img)
# Waits indefinitely until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows





