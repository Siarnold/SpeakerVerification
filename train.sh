#!/bin/sh
# train the spv_cnn end-to-end
# Copyleft(c), Siarnold, 2017
./data_process.py
./tvtlist_produce.py
./spv_cnn_trainvf.py
