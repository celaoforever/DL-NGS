import math
import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
import random
from keras.preprocessing import sequence
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten, Conv1D, MaxPooling1D,LSTM,Bidirectional,BatchNormalization 
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


input_file=sys.argv[1]
p_threshold=0.005
pctg_validate=0.2
pctg_test=0.05
mode="TRAIN" #"PREDICT" #TRAIN


def codeSeq(seq,length):
  res=[]
  #A,T,G,C -> 0,1,2,3
  for char in seq:
    if char == "A" or char == "0" or char == "a":
      res.append([1,0,0,0])
      continue
    if char == "T" or char == "1" or char == "t":
      res.append([0,1,0,0])
      continue
    if char == "G" or char == "2" or char == "g":
      res.append([0,0,1,0])
      continue
    if char == "C" or char == "3" or char == "c":
      res.append([0,0,0,1])
      continue
    res.append([0,0,0,0])
  while len(res)<length:
    res.append([0,0,0,0])
  return res

def expand(input_seq,length):
  res=[]
  for err,cor,err_type,marked,base_change in input_seq:
    res.append([codeSeq(err,length),1,len(err),err_type,marked,base_change])
    res.append([codeSeq(cor,length),0,len(cor),"none","none","none"])
  random.shuffle(res)
  return zip(*res)

def get_err_direction(f,r,b):
  f=int(f)
  r=int(r)
  b=int(b)
  res=""
  if f > 0:
    res+="f"
  if r > 0:
    res+="r"
  if b > 0:
    res+="b"
  return res

def load_data(input_file,p_threshold,pctg_validate,pctg_test):
  seq_len_cutoff=1#0.95
  seq_len_dist=[]
  matched_seq=[]
  for line in open(input_file,'r'):
    line=line.rstrip().split("\t")
    if line[0] == "seq_len":
      continue
    if float(line[2]) >p_threshold:
      continue 
    seq_len_dist.append(int(line[0]))
    matched_seq.append([line[5],line[7],get_err_direction(line[8],line[9],line[10]),line[14],line[15]])
  total_case=len(seq_len_dist)
  seq_len_dist=sorted(seq_len_dist) 
  rank=seq_len_cutoff*total_case
  rank=len(seq_len_dist)-1 if int(rank) >=len(seq_len_dist) else int(rank)
  len_thres=seq_len_dist[int(rank)]
  refined_matched_seq=[]
  for err,cor,err_type,marked,base_change in matched_seq:
    if len(err) <= len_thres:
      refined_matched_seq.append([err,cor,err_type,marked,base_change])
  total_case=len(refined_matched_seq)
  validate_case=int(total_case*pctg_validate)
  test_case=int(total_case*pctg_test)
  random.shuffle(refined_matched_seq)
  validate_matched=refined_matched_seq[0:validate_case]
  test_matched=refined_matched_seq[validate_case:(validate_case+test_case)]
  train_matched=refined_matched_seq[(validate_case+test_case):]
  validate_x,validate_y,validate_seq_len,validate_err_type,validate_x_marked,validate_base_change=expand(validate_matched,len_thres)
  test_x,test_y,test_seq_len,test_err_type,test_x_marked,test_base_change=expand(test_matched,len_thres)
  train_x,train_y,train_seq_len,train_err_type,train_x_marked,train_base_change=expand(train_matched,len_thres)
  return len_thres,np.array(train_x),np.array(train_y),np.array(train_err_type),np.array(validate_x),np.array(validate_y),np.array(validate_err_type),np.array(test_x),np.array(test_y),np.array(test_err_type),np.array(train_x_marked),np.array(train_base_change),np.array(validate_x_marked),np.array(validate_base_change),np.array(test_x_marked),np.array(test_base_change)

def getBase(base):
  try:
    pos=list(base).index(1)
    if pos == 0:
      return "A"
    if pos == 1:
      return "T"
    if pos == 2:
      return "G"
    if pos == 3:
      return "C"
    return "N"
  except:
    return "N"

def decode(seq,start,end):
  res=""
#  decode seq[start,end) and return
  for i in xrange(start,end):
    res+=getBase(seq[i])
  return res.rstrip("N")

def pad_data(train_x,validate_x,test_x,maxlen):
  train_x=sequence.pad_sequences(train_x, maxlen=maxlen)
  validate_x=sequence.pad_sequences(validate_x, maxlen=maxlen)
  test_x=sequence.pad_sequences(test_x,maxlen=maxlen)
  return train_x,validate_x,test_x

def build_model(seq_max_len,kernel_number,kernel_length):
  model = Sequential()
  model.add((Conv1D(kernel_number,kernel_length,activation='relu',input_shape=(seq_max_len,4))))
  model.add(Dropout(0.1))
  model.add(Bidirectional(LSTM(32, dropout=0.1, recurrent_dropout=0.1,return_sequences=True)))
  model.add(Bidirectional(LSTM(32, dropout=0.1, recurrent_dropout=0.1,return_sequences=True)))
  model.add(Bidirectional(LSTM(32, dropout=0.1, recurrent_dropout=0.1)))
  model.add(Dense(1))
  model.add((Activation('sigmoid')))
  model.compile(loss="binary_crossentropy", metrics=["binary_accuracy"],optimizer='adam')
  return model

batch_size=64
epochs=100
kernel_number=64
kernel_length=16
max_len,train_x,train_y,train_err_type,validate_x,validate_y,validate_err_type,test_x,test_y,test_err_type,train_x_marked,train_base_change,validate_x_marked,validate_base_change,test_x_marked,test_base_change=load_data(input_file,p_threshold,pctg_validate,pctg_test)
model=build_model(max_len,kernel_number,kernel_length)
if mode == "TRAIN":
  checkpointer = ModelCheckpoint(filepath="model_param.hdf5", verbose=1, save_best_only=True)
  earlystopper = EarlyStopping(monitor='val_binary_accuracy', patience=10, verbose=1)
  model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(validate_x, validate_y),callbacks=[checkpointer,earlystopper])
  model.load_weights("model_param.hdf5")
  tresults = model.evaluate(test_x,test_y)
  print tresults
elif mode == "PREDICT":
  all_in_x=np.concatenate([train_x,validate_x,test_x],axis=0)
  all_in_y=np.concatenate([train_y,validate_y,test_y],axis=0)
  all_in_x_marked=np.concatenate([train_x_marked,validate_x_marked,test_x_marked],axis=0)
  all_in_base_change=np.concatenate([train_base_change,validate_base_change,test_base_change],axis=0)
  all_in_err_type=np.concatenate([train_err_type,validate_err_type,test_err_type],axis=0)
  model.load_weights("model_param.hdf5")
  all_in_predict=model.predict(all_in_x)
  f=open("predict_res.txt",'w')
  for x,y,e,p in zip (all_in_x,all_in_y,all_in_err_type,all_in_predict):
    r=1 if p >=0.5 else 0
    f.write(decode(x,0,len(x))+"\t"+str(p[0])+"\t"+str(r)+"\t"+str(y)+"\t"+e+"\n")
  f.close()
