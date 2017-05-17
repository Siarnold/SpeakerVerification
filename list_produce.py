#!/usr/bin/env python
# Produce a train list.
# Copyleft(c), Siarnold, 2017

import os

cwd = os.getcwd() # current working directory
dpath = cwd + '/data/prcsd.ignore/dats/' # data path
tname = cwd + '/data/prcsd.ignore/train_list.txt' # train text path

if not os.path.isfile(tname):
	f = open(tname,'w')
	for x in os.listdir(dpath):
		if x[0:3] == 'qys':
			f.write(dpath + x + ' 0\n')
		else:
			f.write(dpath + x + ' 1\n')
	f.close()
