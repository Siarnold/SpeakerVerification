#!/usr/bin/env python
# Produce a train and validation list.
# Copyleft(c), Siarnold, 2017

import os
from random import randint

cwd = os.getcwd() # current working directory
dpath = cwd + '/data/prcsd.ignore/dats/' # data path
tname = cwd + '/data/prcsd.ignore/train_list.txt' # train text path
vname = cwd + '/data/prcsd.ignore/val_list.txt' # validation text path

if not os.path.isfile(tname):
	tf = open(tname, 'w')
	vf = open(vname, 'w')
	for x in os.listdir(dpath):
		rd = randint(0, 3)# generate random integers 0,1,2,3
		# set the train:val to be 3:1		
		if rd < 3:
			f = tf
		else:
			f = vf
		if x[0:3] == 'qys':
			f.write(dpath + x + ' 0\n')
		else:
			f.write(dpath + x + ' 1\n')
	f.close()
