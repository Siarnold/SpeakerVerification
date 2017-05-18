#!/usr/bin/env python
# Test spv_cnn on audios
# Copyleft(c), Siarnold, 2017

import os
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from spv_cnn import spv_cnn, spv_cnn_v1

cwd = os.getcwd() # current working directory
mpath = cwd + '/modelv0-tv.ignore/spv_v0.199-0.04.hdf5' # model path
apath = cwd + '/data/prcsd.ignore/dats/qys_english_0010.dat' # audio segment path
# apath = cwd + '/file.ignore/break/dats/test_0020.dat'
nClasses = 2

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
model = spv_cnn(nDim = 16000, nClass = nClasses)
model.load_weights(mpath)
x = np.fromfile(apath, dtype = np.uint16) # x is (16000,) matrix
x = np.expand_dims(x, axis=0)
x = np.expand_dims(x, axis=2)
pred = model.predict(x, batch_size=1, verbose=1)
# pred is an ndarray (number of batches) of ndarrays
pred = pred[0]
if pred.argmax() == 0:
	print('Yes! QYS is speaking.\n')
else:
	print('Sorry, you are not successfully verified.\n')

