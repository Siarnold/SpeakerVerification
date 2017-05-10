#!/usr/bin/env python
# FFT the audio data to derive its frequent field feature.
# Copyleft(c), Siarnold, 2017

import os
import wave

fpath = './data/ty/'# file path
pfpath = './data/pty/'# processed file path
# isdir/isfile/exists
if not os.path.isdir(pfpath):
	os.mkdir(pfpath)
	for fname in os.listdir(fpath):# file name
		audif = fpath + fname
		audipf = pfpath + fname[0:-4] + '.wav'
		print(audipf)
		str = 'ffmpeg -i ' + audif + ' -ar 2000 ' + audipf
		os.system(str)

# the audio file
# audif = filepath + filename
# audi = wave.open(audif)
# print(audi.getframerate(), audi.getnframes())
