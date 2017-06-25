#!usr/bin/env python
# count the frames and time of the dataset
# Copyleft(c), Siarnold, 2017

import os
import wave
import numpy as np

fPath = './data/raw/'
cPath = './data/count.ignore/' # count path
nFrame = 0
time = 0
if not os.path.isdir(cPath):
	os.mkdir(cPath)
	for name in os.listdir(fPath):
		audif = fPath + name
		audicf = cPath + name[0:-4] + '.wav'
		cstr = 'ffmpeg -i ' + audif + ' ' + audicf
		os.system(cstr)

for name in os.listdir(cPath):
	audi = wave.open(cPath + name)
	nframes = audi.getnframes()
	sf = audi.getframerate()
	time += nframes/sf
	nFrame += nframes

print(nFrame, time)
