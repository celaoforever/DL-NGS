# DL-NGS
An unsupervised and universal deep learning based framework for sequencing error profiling and correcting

## File format for input data, each column stands for:
* seq_len           : sequence length
* gc                : GC-content
* p_min             : minimum p-value of mismatch rate in this region       
* err_pos           : the position sequencing error occurs
* marked_err_seq    : sequences with errors marked
* err_seq           : sequences with errors
* matched_pos       : position of GC matched error-free sequences
* matched_seq       : sequences of GC matched error-free sequences
* fwd_err           : if contains errors in the forward strand
* rvs_err           : if contains errors in the reverse strand   
* both_err          : if contains errors in both the strands
* depth_err         : average sequencing depth of error sequence
* depth_cor         : average sequencing depth of error-free sequence
* comment           : comment
* marked_seq        : html marked error sequences
* err_type          : base changes of each error position

## prerequisites
* python 2, keras (https://keras.io/)

## How to run
* There are two modes for the script "train_and_predict.py". One is train mode and the other is predict mode. 
* Train mode will training the model based on the input data and output the model parameters into file "model_param.hdf5".
* Predict mode will load an existing model, predict the sequences in the input file, and output the prediction to file "predict_res.txt".
* The input file format for train and predict mode is the same.
* To switch between train mode and predict mode, modify line 21 in the script and set the variable "mode": mode="TRAIN" or "PREDICT".
* When in train mode, percentage of validation datasets can be specified by modifying variable "pctg_validate" (defualt is 0.2). Percentage of test datasets can be specified by modifying "pctg_test" (default is 0.05).
* After setting all the parameters, run: python train_and_predict.py input_data.txt

