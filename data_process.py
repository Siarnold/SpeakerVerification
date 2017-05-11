#!/usr/bin/env python
# FFT the audio data to derive its frequent field feature.
# Copyleft(c), Siarnold, 2017

import os
import wave
import numpy as np

fpath = './data/ty/' # file path
pfpath = './data/pty/' # processed file path
samRate = 4000 # sample rate
samTime = 4 # sample time

pfpath1 = pfpath + 'wavs/'
pfpath2 = pfpath + 'dats/'
nSamFrame = samRate * samTime # number of sample frames
# isdir/isfile/exists
# convert formats to wave
if not os.path.isdir(pfpath):
	os.mkdir(pfpath)
	os.mkdir(pfpath1)
	os.mkdir(pfpath2)
	for fname in os.listdir(fpath): # file name
		audif = fpath + fname
		audipf = pfpath1 + fname[0:-4] + '.wav'
		# convert format with sample rate and channel 1
		cstr = 'ffmpeg -i ' + audif + ' -ar ' + str(samRate) + ' -ac 1 ' + audipf # command string
		os.system(cstr)

# split the wave files and save as .dat files
for wname in os.listdir(pfpath1): # wave name
	audi = wave.open(pfpath1 + wname) # audio
	nframes = audi.getnframes()
	sdata = audi.readframes(nframes) # string type of data
	audi.close()
	wdata = np.fromstring(sdata, dtype = np.short) # short-type data, 2 Bytes
	nsplits = nframes // 16000 # number of valid splits (with 16000 frames), whether the remaider is 0 or else
	splits = np.split(wdata, [16000 * x + 1 for x in range(1, nsplits + 1)])
	for x in range(nsplits):
		split = splits[x]
		split = np.fft.fft(split) # FFT to extract raw feature
		split.tofile(pfpath2 + wname[0:-4] + '_{:0>4d}.dat'.format(x))
		print(split.shape)

# wdata = np.fromfile(pfpath2 + wname[0:-4] + '.dat', dtype = np.short)

