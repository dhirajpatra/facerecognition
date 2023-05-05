import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as pl
import pandas as pd
from PIL import Image


# Set font type, font scale, color, and thickness
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_color = (0, 0, 255)  # BGR color format
font_thickness = 3

# This module contains all common functions that are called in tester.py file


# Given an image below function returns rectangle for face detected along with gray scale image
def faceDetection(test_img):
    # convert color image to grayscale
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    # Load haar classifier
    harr_cascade_model = "HaarCascade/haarcascade_frontalface_default.xml"
    # harr_cascade_model = "HaarCascade/haarcascade_frontalface_alt.xml"
    # harr_cascade_model = cv2.data.haarcascades + 'haarcascade_frontalface.xml'
    face_haar_cascade = cv2.CascadeClassifier(harr_cascade_model)
    # detectMultiScale returns rectangles
    faces = face_haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.32, minNeighbors=5
    )

    return faces, gray_img


# to calculate euclidean distance
def trignometry_for_distance(a, b):
    return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) + ((b[1] - a[1]) * (b[1] - a[1])))


# Find eyes
def faceAlignment(img_path):
    """
    Given an image path, this function aligns the face in the image.
    """
    img, gray_img = faceDetection(img_path)
    img_raw = img_path.copy()

    path_for_eyes = "HaarCascade/haarcascade_eye.xml"
    eye_detector = cv2.CascadeClassifier(path_for_eyes)
    eyes = eye_detector.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

    # For multiple people in an image find the largest pair of eyes
    if len(eyes) >= 2:
        eye = eyes[:, 2]
        container1 = []
        for i in range(0, len(eye)):
            container = (eye[i], i)
            container1.append(container)
        df = pd.DataFrame(container1, columns=["length", "idx"]).sort_values(by=["length"])
        eyes = eyes[df.idx.values[0:2]]

        # Deciding to choose left and right eye
        eye_1 = eyes[0]
        eye_2 = eyes[1]
        if eye_1[0] > eye_2[0]:
            left_eye = eye_2
            right_eye = eye_1
        else:
            left_eye = eye_1
            right_eye = eye_2

        # Center of eyes
        # Center of right eye
        right_eye_center = (
            int(right_eye[0] + (right_eye[2] / 2)),
            int(right_eye[1] + (right_eye[3] / 2)),
        )
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]

        # Center of left eye
        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)),
            int(left_eye[1] + (left_eye[3] / 2)),
        )
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]

        # finding rotation direction
        if left_eye_y > right_eye_y:
            print("Rotate image to clock direction")
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate image direction to clock
        else:
            print("Rotate to inverse clock direction")
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock

        # for x2, y2, w2, h2 in eyes:
        #     radius = int(round((w2 + h2) * 0.25))
        #     img = cv2.circle(img, left_eye_center, radius, (0, 0, 255), 4)

        a = trignometry_for_distance(left_eye_center, point_3rd)
        b = trignometry_for_distance(right_eye_center, point_3rd)
        c = trignometry_for_distance(right_eye_center, left_eye_center)
        if b * c != 0:
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = (np.arccos(cos_a) * 180) / math.pi
        else:
            angle = 0

        if direction == -1:
            angle = 90 - angle
        else:
            angle = -(90 - angle)

        # rotate image
        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle))

        return new_img, gray_img
    return img, gray_img


# Given a directory below function returns part of gray_img which is face along with its label/ID
def labels_for_training_data(directory, name):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")  # Skipping files that startwith .
                continue

            id = os.path.basename(path)  # fetching subdirectory names eg. 0

            # check if all folder need to check or not as per tester setting
            if name[int(id)] == "":
                continue
            img_path = os.path.join(path, filename)  # fetching image path
            print("img_path:", img_path)
            print("id:", id)
            test_img = cv2.imread(img_path)  # loading each image one by one
            if test_img is None:
                print("Image not loaded properly")
                continue

            # Calling faceDetection function to return faces detected in particular image
            # faces_rect, gray_img = faceDetection(test_img)
            faces_rect, gray_img = faceAlignment(test_img)
            if len(faces_rect) != 1:
                continue  # Since we are assuming only single person images are being fed to classifier
            (x, y, w, h) = faces_rect[0]
            # cropping region of interest i.e. face area from grayscale image
            roi_gray = gray_img[y : y + w, x : x + h]
            faces.append(roi_gray)
            faceID.append(int(id))
            print(faceID)
    return faces, faceID


# Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces, faceID):
    """
    Trains a face recognition model using the given faces and face IDs.

    Args:
    - faces (list of numpy arrays): A list of face images.
    - faceID (list of integers): A list of integers representing the IDs of the faces.

    Returns:
    - face_recognizer (cv2.face_LBPHFaceRecognizer object): A trained face recognizer.
    """
    # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer = cv2.face.LBPHFaceRecognizer.create()
    # face_recognizer = cv2.face.createLBPHFaceRecognizer()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer


# Below function draws bounding boxes around detected face in image
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)


# Below function writes name of person for detected label
def put_text(test_img, text, x, y):
    cv2.putText(
        test_img, text, (x, y - 5), font, font_scale, font_color, font_thickness
    )
