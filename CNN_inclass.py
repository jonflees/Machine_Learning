#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:29:07 2020

@author: jonflees
"""


from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
# Plots first image in dataset

print(X_train)
plt.imshow(X_train[0])

#check image shape
X_train[0].shape

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

from keras.utils import to_categorical
# one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train[0]



#### BUILDING THE MODEL

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

#create model
model = Sequential()

#add more layers
model.add(Conv2D(64, kernel_size=3, activation='relu',  input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

print("compile model using accuracy to measure model performance")
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

print(" Train the model")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#make predictions on first 4 images
model.predict(X_test[:4])

#compare to actuals
y_test[:4]