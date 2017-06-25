# A starting research into the applications of signals and systems.
This project tests the ability of CNN on the task of speaker verification, as an assignment for the course of Signals and Systems in Tsinghua University in the spring of 2017.
------
## Train the CNN
* The training could be started simply by running train.sh
## Files in this repository
* SS.pdf is the essay for this assignments.  
  Other pdf files are the referenced literature in this research. 
* data_process.py is used to preprocess the data used in the CNN.  
  data_process_for_open_test.py is used for preprocess data out of the dataset.  
  dataCount.py is used to count the time and frames of the dataset.
* list_produce.py / tvlist_produce.py / tvtlist_produce.py are used to produce lists / train-val lists / train-val-test lists for the CNN.
* spv(= Speaker Verification)_cnn.py is used to formulate the network under keras, in which 3 models are included.
* spv_cnn_trainv0.py / spv_cnn_trainv1.py / spv_cnn_trainvf.py train the 3 models respectively.
* spv_cnn_test.py is used to test one clip of processed data.  
  spv_cnn_batch_test.py tests the data in test list.  
  spv_cnn_open_batch_test tests a batch of processed data from out of the dataset.
------
FINISHED on June 25th, 2017
