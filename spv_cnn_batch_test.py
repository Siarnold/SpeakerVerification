#!/usr/bin/env python
# Test spv_cnn on audios
# Copyleft(c), Siarnold, 2017

import os
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from spv_cnn import spv_cnn, spv_cnn_v1, spv_cnn_vf

cwd = os.getcwd() # current working directory
# model path
mpath = cwd + '/modelvf.ignore/spv_vf.299-0.00.hdf5'
# test list path
tslist = cwd + '/data/prcsdv3.ignore/test_list.txt'
nClasses = 2
totalCount = 0 # count the total verification
bingoCount = 0 # count the correct verification

def get_session(gpu_fraction=0.9):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


KTF.set_session(get_session())
model = spv_cnn_vf(nDim = 16000, nClass = nClasses)
model.load_weights(mpath)

f = open(tslist)
for line in f:
	x = line.strip()
	x = x.split(' ')
	y = x[-1]
	x = x[0]
	x = np.fromfile(x, dtype = np.uint16) # x is (16000,) matrix
	x = np.expand_dims(x, axis=0)
	x = np.expand_dims(x, axis=2)
	pred = model.predict(x, batch_size=1, verbose=1)
	# pred is an ndarray (number of batches) of ndarrays
	pred = pred[0]
	totalCount = totalCount + 1
	
	if pred.argmax() == int(y) :
		bingoCount = bingoCount + 1

print('The accuracy rate is: ', 1.0 * bingoCount / totalCount)


