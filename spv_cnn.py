#!/usr/bin/env python
# Build the structure of a CNN for speaker verification.
# Copyleft(c), Siarnold, 2017

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D

def spv_cnn(nDim = 16000, nClass = 2):
	model = Sequential()
	model.add(Conv1D(32, 5, activation='relu', input_shape=(nDim, 1), name='block1_conv1'))
	model.add(Conv1D(4, 5, activation='relu', name='block1_conv2'))
	model.add(Dropout(0.25, name='block1_drop1'))
	model.add(Flatten(name='block1_flat1'))
	model.add(Dense(128, activation='relu', name='block2_dense1'))
	model.add(Dropout(0.5, name='block2_drop1'))
	model.add(Dense(nClass, activation='softmax', name='predict'))
	return model

def spv_cnn_v1(nDim = 16000, nClass = 2):
	model = Sequential()
	model.add(Conv1D(32, 5, activation='relu', input_shape=(nDim, 1), name='block1_conv1'))
	model.add(Conv1D(4, 5, activation='relu', name='block1_conv2'))
	model.add(Dropout(0.25, name='block1_drop1'))
	model.add(Flatten(name='block1_flat1'))
	model.add(Dense(1024, activation='relu', name='block2_dense1'))
	model.add(Dropout(0.5, name='block2_drop1'))
	model.add(Dense(1024, activation='relu', name='block2_dense2'))
	model.add(Dropout(0.5, name='block2_drop2'))
	model.add(Dense(nClass, activation='softmax', name='predict'))
	return model

def spv_cnn_vf(nDim = 16000, nClass = 2):
	model = Sequential()
	model.add(Conv1D(3, 3, activation='relu', input_shape=(nDim, 1), name='block1_conv1'))
	model.add(Conv1D(3, 3, activation='relu', name='block1_conv2'))
	model.add(Dropout(0.25, name='block1_drop1'))
	model.add(Flatten(name='block1_flat1'))
	model.add(Dense(64, activation='relu', name='block2_dense1'))
	model.add(Dropout(0.5, name='block2_drop1'))
	model.add(Dense(nClass, activation='softmax', name='predict'))
	return model

if __name__ == "__main__":
	model = spv_cnn_vf()
	model.summary()
