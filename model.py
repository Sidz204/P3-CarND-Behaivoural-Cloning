import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten,Dense, Activation, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

lines = []

with open('data/driving_log.csv') as csvfile:   #reading csv file
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


#reading images from the csv read above
		
images = []
angles = []
for line in lines:
	c=0
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'data/IMG/' + filename
		image = cv2.imread(current_path)
		
		if image is not None:
			image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  #converting to RGB colorspace
			images.append(image)
		else:
			c = 1
			break
	if(c==1):
		continue
	else:
		correction = 0.2
		angle = float(line[3])
		angles.append(angle)
		angles.append(angle+correction)      #including correction for left and right camera images
		angles.append(angle-correction)



#adding flipped images
aug_images = []
aug_angles = []
for image,angle in zip(images,angles):
	aug_images.append(image)
	aug_angles.append(angle)
	if(angle != 0.0 or angle!= -0.0):     #when angle is 0.0 no need to add flipped image of that image
		flipped_image = cv2.flip(image,1)
		flipped_angle = angle * -1.0
		aug_images.append(flipped_image)
		aug_angles.append(flipped_angle)




	
X_train = np.array(aug_images) #all images and angles are converted into numpy array
y_train = np.array(aug_angles)



print(X_train.shape)


#model architecture

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))    #normalizing the data
model.add(Cropping2D(cropping = ((70,20),(0,0))))                          #cropping images
model.add(Convolution2D(6,5,5,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(32,3,3,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=2)
model.save('model.h5')
