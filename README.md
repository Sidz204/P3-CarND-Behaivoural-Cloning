# P3-CarND-Behaivoural-Cloning

### Overview

In this project, deep neural networks and convolutional neural networks is used to clone driving behavior. A model is trained, validated and tested using Keras. The model will output a steering angle to an autonomous vehicle. A simulator is provided where we can steer a car around a track for data collection. Image data and steering angles are used to train a neural network and then this model is used to drive the car autonomously around the track in the simulator.


### Project Files

- model.py  : script used to create and train the model
- drive.py  : script to drive the car - feel free to modify this file
- model.h5  : a trained Keras model
- writeup.md: a report writeup file
- video.mp4 : a video recording of your vehicle driving autonomously around the track for at least one full lap


### Instructions to run the code:

- while running on AWS instance please activate carnd-term1 environment and in that run "pip install opencv-python" if import cv2 gives error
- Use scp command to copy any files from local machine to AWS instance & vice versa and scp -rv for entire directory
- Use command 'python drive.py model.h5 run1' to save the images scene by the agent in 'run1 directory and 'python video.py run1' to create videos based on images in run1 directory.
