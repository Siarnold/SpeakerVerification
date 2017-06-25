#!/usr/bin/env python
# Train the spv_cnn
# Copyleft(c), Siarnold, 2017

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np
from spv_cnn import spv_cnn, spv_cnn_v1, spv_cnn_vf

# set parameters
batchSize = 16
nClasses = 2
nEpochs = 300

# the paths
cwd = os.getcwd() # current working directory
mpath = cwd + '/modelvf.ignore/' # model path
mname = mpath + 'spv_vf.{epoch:03d}-{val_loss:.2f}.hdf5'
lname = mpath + 'spv_vf_training.log'

tpath = cwd + '/data/prcsdv3.ignore/train_list.txt' # train text path
vpath = cwd + '/data/prcsdv3.ignore/val_list.txt' # validation text path
dpath = cwd + '/data/prcsdv3.ignore/dats/' # data path

# count the number of samples
f = open(tpath)
nTrain = len(f.readlines()) # number of train samples
f.close()
f = open(vpath)
nVal =  len(f.readlines())# number of val samples
f.close()

def get_session(gpu_fraction=0.9):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def generate_fit(path, batchSize, nClasses):
	while True:
		cnt = 0
		X = []
		Y = []
		f = open(path)
		for line in f:
			x = line.strip() # remove the leading and trailing whilespace
			x = x.split(' ') # split to a list like ['*dat', '0']
			y = x[-1]
			x = x[0]
			x = np.fromfile(x, dtype = np.uint16) # x is (16000,) matrix
			# print(x.max(),x.min(),x.mean())
			x = np.expand_dims(x, axis=2)
			# one-hot encoding			
			z = np.zeros(nClasses)
			z[int(y)] = 1
			X.append(x)
			Y.append(z)
			cnt += 1

			if cnt == batchSize:
				cnt = 0
				X0 = np.array(X)
				Y0 = np.array(Y)
				X = []
				Y = []
				yield({'block1_conv1_input':X0}, {'predict':Y0})
		f.close()


if __name__ == '__main__':
	KTF.set_session(get_session())
	model = spv_cnn_vf(nDim = 16000, nClass = nClasses)
	model.summary()
	sgd = SGD(lr=1e-6, decay=3e-9, momentum=0.9, nesterov=True)
	# model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	
	if not os.path.isdir(mpath):
		os.mkdir(mpath)

	# checkpoint
	cp = ModelCheckpoint(mname, save_weights_only=True, period=10)
	# csv logger
	cl = CSVLogger(lname)
	model.fit_generator(generate_fit(tpath, batchSize, nClasses), 
		validation_data=generate_fit(vpath, batchSize, nClasses), 
		steps_per_epoch=np.ceil(nTrain/batchSize), 		
		epochs=nEpochs, validation_steps=nVal, 
		verbose=1, callbacks=[cp, cl])

