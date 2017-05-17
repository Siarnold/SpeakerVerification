#!/usr/bin/env python
# FFT the audio data to derive its frequent field feature.
# 	Effective FFT confirmed with uint16.
# Copyleft(c), Siarnold, 2017

import os
import wave
import numpy as np

fpath = './data/raw/' # file path
# use .ignore to avoid uploading
pfpath = './data/prcsd.ignore/' # processed file path
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
		wdata = np.fromstring(sdata, dtype = np.uint16)
		# 0 - 65535 effectively convert
		# print(wdata.mean()) # mean around 33000
		nsplits = nframes // nSamFrame # number of valid splits (with 16000 frames), whether the remaider is 0 or else
		splits = np.split(wdata, [nSamFrame * x for x in range(1, nsplits + 1)])
		for x in range(nsplits):
			split = splits[x]
			split = np.fft.fft(split) # FFT to extract raw feature
			# split = np.absolute(split)
			# print(split.max(),split.min(), split.mean())
			# if not //, max = 5e8 min = 1e-4
			split = (np.absolute(split) // 10000).astype(np.uint16)
			# print(split.max(),split.min())
			split.tofile(pfpath2 + wname[0:-4] + '_{:0>4d}.dat'.format(x))
			# split = np.fromfile(pfpath2 + wname[0:-4] + '_{:0>4d}.dat'.format(x), dtype = np.uint16)

