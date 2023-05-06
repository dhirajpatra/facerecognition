import argparse
import sys
import time
import os
import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


# Constants
TARGET_SIZE = (256, 256)
MODEL_BASE_PATH = "/home/pi/dust_detection/"
MAX_RESULTS = 1
SCORE_THRESHOLD = 0.7
FPS_AVG_FRAME_COUNT = 10
TEXT_COLOR = (0, 0, 255)  # red
FONT_SIZE = 1
FONT_THICKNESS = 1


def run(model_path, camera_id, width, height, num_threads, enable_edgetpu):
    """Continuously run inference on images acquired from the camera.

    Args:
      model_path: Path of the TFLite object detection model.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
      num_threads: The number of CPU threads to run the model.
      enable_edgetpu: True/False whether the model is an EdgeTPU model.
    """

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Initialize the image classification model
    base_options = core.BaseOptions(
        file_name=model_path, use_coral=enable_edgetpu, num_threads=num_threads)
    classification_options = processor.ClassificationOptions(
        max_results=MAX_RESULTS, score_threshold=SCORE_THRESHOLD)
    options = vision.ImageClassifierOptions(
        base_options=base_options, classification_options=classification_options)
    classifier = vision.ImageClassifier.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            sys.exit(
                "ERROR: Unable to read from webcam. Please verify your webcam settings."
            )

        counter += 1
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Run inference
        try:
            # Create TensorImage from the RGB image
            tensor_image = vision.TensorImage.create_from_array(rgb_image)
            # List classification results
            categories = classifier.classify(tensor_image)
        except ValueError as e:
            print(f"Error: {e}")
            categories = None

        # print(categories)
        category_name = ''
        class_dict = {0: "Dhiraj", 1: "Om", 2: "Tanushree"}
        cat_details = {}
        for idx, category in enumerate(categories.classifications[0].categories):
            # category_name = category.category_name
            # score = round(category.score, 2)
            score = category.score
            index = category.index
            # print(category)
            cat_details[index] = score

        cat_details = dict(sorted(cat_details.items(), key=lambda x: x[1], reverse=True))
        # print(cat_details)

        output_details = []
        for k, v in cat_details.items():
            output_details.append({"key": k, "score": v})
            category_name = class_dict[k]

        print(output_details)

        # Calculate the FPS
        if counter % FPS_AVG_FRAME_COUNT == 0:
            end_time = time.time()
            fps = FPS_AVG_FRAME_COUNT / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = "FPS = {:.1f}".format(fps)
        text_location = (24, 20)
        cv2.putText(
            image,
            fps_text + ' ' + category_name,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord("q"):
            break
        cv2.imshow("object_detector", image)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Path of the object detection model.",
        required=False,
        default="face_recognition_model_edge.tflite",
    )
    parser.add_argument(
        "--cameraId", help="Id of camera.", required=False, type=int, default=0
    )
    parser.add_argument(
        "--frameWidth",
        help="Width of frame to capture from camera.",
        required=False,
        type=int,
        default=640,
    )
    parser.add_argument(
        "--frameHeight",
        help="Height of frame to capture from camera.",
        required=False,
        type=int,
        default=480,
    )
    parser.add_argument(
        "--numThreads",
        help="Number of CPU threads to run the model.",
        required=False,
        type=int,
        default=4,
    )
    parser.add_argument(
        "--enableEdgeTPU",
        help="Whether to run the model on EdgeTPU.",
        action="store_true",
        required=False,
        default=False,
    )
    args = parser.parse_args()

    run(
        args.model,
        int(args.cameraId),
        args.frameWidth,
        args.frameHeight,
        int(args.numThreads),
        bool(args.enableEdgeTPU),
    )


if __name__ == "__main__":
    main()
