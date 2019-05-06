# Face Recognition

Face Recognition using OpenCV in Python

### Prerequisites

Numpy</br>
OpenCV


### Installing

Check your python is 3.7 otherwise update and make default this version

Install Numpy via anaconda:
conda install numpy

Install OpenCV via anaconda:
conda install -c menpo opencv


## Running the tests

Run Tester.py script on commandline to train recognizer on training images and also predict test_img:<br>
`python tester.py`

1. Run: `python videotoimg.py`  
[to capture your frame images from video, it will automatically stop after taking 99 images. To generate test images for training classifier use videoimg.py file.</br>]

2. Place Images for training the classifier in trainingImages folder.If you want to train clasifier to recognize multiple people then add each persons folder in separate label markes as 0,1,2,etc and then add their names along with labels in tester.py/videoTester.py file in 'name' variable.</br>
Now cut all images from "CapturedImages" folder to "trainingImages/id" folder.

3. Run: `python resizeImages.py` 
[to resize all trainingImages as per the standard size, make sure that all id eg. 0 sub folders are already there in "resizedTrainingImages"]
 
4. Place some test images in TestImages folder that you want to predict  in tester.py file. Put single prominent images for each candidate.

5. To do test run via tester.py give the path of image in test_img variable</br>

6. Use "videoTester.py" script for predicting faces realtime via your webcam.(But ensure that you run tester.py first since it generates training.yml file that is being used in "videoTester.py" script.


## Acknowledgments
* https://www.superdatascience.com/opencv-face-recognition/
* https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/

