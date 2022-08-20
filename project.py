#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:31:21 2020

@author: jonflees
"""


#from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from skimage.transform import resize
from skimage.io import imread_collection
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


X_train = []
y_train = []
X_test = []
y_test = []
y_counter = 0

def addToX(col_dir):
        #gets all photos from folder
    col = imread_collection(col_dir)
    counter = 0

    for i in col:
           #to  resize the data and add to training data
        i_resized = resize(i, (256,256,3)) 
        X_train.append(i_resized)
        y_train.append(y_counter)
        print(counter)
        counter+=1
        
def addToTestX(col_dir):
        #gets all photos from folder
    col = imread_collection(col_dir)
    counter = 0

    for i in col:
           #to  resize the data and add to training data
        i_resized = resize(i, (256,256,3)) 
        X_test.append(i_resized)
        print(counter)
        counter+=1

addToX('AmericanFlag/*.jpg')
y_counter+=1
addToX('RussianFlag/*.jpg')
y_counter+=1
addToX('christmas trees/*.jpg')
y_counter+=1
addToX('santa hats/*.jpg')
y_counter+=1
addToX('italian flag/*.jpg')
y_counter+=1
addToX('hanukkah/*.jpg')
y_counter+=1
addToX('xmas dec/*.jpg')
y_counter+=1
addToX('Cross/*.jpg')
y_counter+=1
addToX('canadianFlag/*.jpg')

addToTestX('test_test/*.jpg')


print("X size: ", len(X_train))  
print("Y size: ", len(y_train))  

#print(X_train[437])
#print(y_train[437])

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test =  np.asarray(X_test)


print(X_train[300].shape)
y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)



model = Sequential()

#add more layers
model.add(Conv2D(64, kernel_size=3, activation='relu',  input_shape=(256,256,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(9, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

print()
print("*** Training the model ***")
model.fit(X_train, y_train, epochs=4)

print(model.predict(X_test[:5]))
num_counter = 0
for i in model.predict(X_test[:100]):
   y_test.append(i)
   num_counter+=1  
   

    
print(y_test)
num_counter2 = 0
for r in y_test:
    num_counter3 = 0
    for s in r:
        num_counter3+=1
        if s >= .5:
            print("Pic Number: ", num_counter2, " with ",s, " at position ", num_counter3)
            plt.imshow(X_test[num_counter2])
    num_counter2+=1    
    