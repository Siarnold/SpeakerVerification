#!/usr/bin/env python
# Produce train, validation and test list.
# Copyleft(c), Siarnold, 2017

import os
from random import randint

cwd = os.getcwd() # current working directory
dpath = cwd + '/data/prcsdv3.ignore/dats/' # data path
tname = cwd + '/data/prcsdv3.ignore/train_list.txt' # train text path
vname = cwd + '/data/prcsdv3.ignore/val_list.txt' # validation text path
tsname = cwd + '/data/prcsdv3.ignore/test_list.txt' # test text path
tstr1 = 'lcd_chinese_pre'
tstr2 = 'qys_english'
tstr3 = 'ty_german'

if not os.path.isfile(tname):
	tf = open(tname, 'w')
	vf = open(vname, 'w')
	tsf = open(tsname, 'w')
	for x in os.listdir(dpath):
		if x[0:15] == tstr1 or x[0:11] == tstr2 or x[0:9] == tstr3:
			f = tsf # the test list
		else:
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
