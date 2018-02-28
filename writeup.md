# **Behavioral Cloning** 

## Overview

### In this project, deep neural networks and convolutional neural networks is used to clone driving behavior. A model is trained, validated and tested using Keras. The model will output a steering angle to an autonomous vehicle. A simulator is provided where we can steer a car around a track for data collection. Image data and steering angles are used to train a neural network and then this model is used to drive the car autonomously around the track in the simulator.

---


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 6 and 64 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Also, shuffling is used.

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road . I augmented the data using flip.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the Lenet model. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I first tried a simple Lenet architecture and found that it didn't give proper results on training and validation data. 

Next, I added few more convolution layers and dense layers similar to NVidia model. And I used Lambda layer to generalize the data using (x/255)-0.5.

My first model  had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added dropouts. Also I cropped the images using model.add(Cropping2D()) to crop the unnecessary noise like trees,sky etc. To recover from being off-center, I used left and right camera images and added a correction of 0.2

Then I found that the car in the simulator was more inclined towards left , so as to generalize the model I flipped the images using cv2.flip().

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I further added training data images in these areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with 4 convolution layers, two with 5x5 filters and two with 3x3 filters each followed by relu activation and maxpooling. After this, there are 5 fully connected dense layers , first two are followed by dropouts with keep probability 0.5.  

Here is a visualization of the architecture

![architecture](/images/model_arch1.JPG)



#### 3. Creation of the Training Set & Training Process

I have used Udacity sample driving data. I also included left and right camera images along with center camera images . Here is an example image of center lane driving and also left and right camera images:

Centre:

![center](/images/center.jpg)



Left :

![left](/images/left.jpg)



Right:

![right](/images/right.jpg)



To augment the data set, I also flipped images and angles thinking that this would help generalize the model so that it will not always inclined towards left or right . For example, here is an image that has then been flipped:

![original](/images/original.png)
![flipped](/images/flipped.png)


After the collection process, I had nearly 35000 images data. I then preprocessed this data by using Lambda layer and Cropping layer. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 but worked well with 2 also. I used an adam optimizer so that manually training the learning rate wasn't necessary.
