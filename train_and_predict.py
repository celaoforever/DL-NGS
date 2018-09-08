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
mode="PREDICT" #TRAIN


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


def sortBaseFreq(freq):
   freqs=zip(freq.keys(), freq.values())
   freqs=sorted(freqs,reverse=True,key=lambda item:item[1])
   return zip(*freqs)

def printSeqs(freq):
  for i in xrange(0,4):
    for item in freq:
      print item[0][i],
    print ""
  for i in xrange(0,4):
    for item in freq:
      print str(item[1][i])+" ",
    print ""

def getFreq(detected):
  #ATGC
  freq=[]
  for base_index in xrange(0,len(detected[0][0])):
    valid=0
    baseDict={"A":0.0,"T":0.0,"G":0.0,"C":0.0}
    for seq_index in xrange(0,len(detected)):
      if detected[seq_index][1]<0.1:
        continue
      valid+=1
      baseDict[detected[seq_index][0][base_index]]+=1
    for base in baseDict:
      baseDict[base]/=float(valid)
    freq.append(sortBaseFreq(baseDict))
  printSeqs(freq)

def findPeakRegion(convolution_res,kernel_len):
  max_value =  max(convolution_res)
  max_index =  convolution_res.index(max_value)
  return max_index,min(len(convolution_res)+kernel_len-1,(max_index+kernel_len)),max_value

def markRegion(seq,start,end):
  res=""
  seq_idx=0
  if start ==0:
    res+="*"
  for i in range(0,len(seq)):
    res+=seq[i]
    if seq[i].upper() in ["A","T","G","C","N"]:
      seq_idx+=1
      if seq_idx == start or seq_idx==end:
        res+="*"
  res=res.replace("*]","]*")
  return res

def getActivatedMotifFromEachKernel(model,input_x,input_err_type,input_predict,kernel_length,marked,base_change):
  conv1_f=K.function([model.input], [model.layers[0].output])
  res=conv1_f([input_x])
  f=open("kernels_add_input_seq_for_0.95depth_add_base_change.txt",'w')
  f.write("kernel\tseq\tmax_value\tlabel\tpredicted\tseq_index\tmarked_seq\tbase_change\n")
  for kernel_index in xrange(0,len(res[0][0][0])):
    kernel_detected=[]
    for input_index in xrange(0,len(res[0])):
      seq_response=[]
      for cursor in xrange(0,len(res[0][0])):
        seq_response.append(res[0][input_index][cursor][kernel_index])
      start,end,max_value=findPeakRegion(seq_response,kernel_length)
      seq=decode(input_x[input_index],start,end)
      marked_region=""
      if marked[input_index] == "none":
        marked_region=markRegion(decode(input_x[input_index],0,len(input_x[input_index])),start,end)
      else:
        marked_region=markRegion(marked[input_index],start,end)
      if len(seq)==kernel_length:
        f.write("kernel "+str(kernel_index)+"\t"+seq+"\t"+str(max_value)+"\t"+str(input_err_type[input_index])+"\t"+str(input_predict[input_index][0])+"\t"+str(input_index)+"\t"+marked_region+"\t"+base_change[input_index]+"\n")

def findValley(x,y):
  res=[]
  for i in range(1,len(y)-1):
    if y[i]<=y[i-1] and y[i] <=y[i+1]:
      res.append([ x[i],y[i]])
  return res

def findPeak(x,y):
  res=[]
  for i in range(1,len(y)-1):
    if y[i]>=y[i-1] and y[i]>=y[i+1]:
      res.append([x[i],y[i]])
  return res

def plotDensity(data,pngfile):
  #random.shuffle(data)
  #data=data[:10000]
  density = gaussian_kde(data)
  xs = np.linspace(min(data),max(data),1000)
  density.covariance_factor = lambda : .25
  density._compute_covariance()
  ys=density(xs)
  mean=np.mean(data)
  valley=findValley(xs,ys)
  peak=findPeak(xs,ys)
  label=""
  label=label + "mean  = "+str("{:5.4f}".format(mean))+"\n"
  for x,y in valley:
    label+= "valley = ["+str("{:5.4f}".format(x))+", "+str("{:5.4f}".format(y))+"]\n"
  for x,y in peak:
    label+= "peak = ["+str("{:5.4f}".format(x))+", "+str("{:5.4f}".format(y))+"]\n"    
  plt.plot(xs,ys,label=label)
  plt.legend(loc="best")
  plt.savefig(pngfile)
  plt.close()

def createBin(data,start,end,bins):
  hist,bin_edges=np.histogram(data,range=(start,end),bins=bins)
  return hist

def isSameDist(data1,data2):
  start=min(data1+data2)
  end=max(data1+data2)
  bins=10
  hist1=createBin(data1,start,end,bins)
  hist2=createBin(data2,start,end,bins)
  chi,p,df,theoretical=chi2_contingency(np.array([hist1,hist2]))
  return p

def getProb(input_dist):
  res=[]
  total=sum(input_dist)
  for i in range(0,len(input_dist)):
    res.append(float(1 if input_dist[i]==0 else  input_dist[i])/float(total))
  return res

def crossEntropy(target_dist,input_dist):
  target_prob=getProb(target_dist)
  input_prob=getProb(input_dist)
  res=0.0
  for i in range(0,len(target_prob)):
    res+=target_prob[i]*math.log(input_dist[i],2)
  return (-1)*res

def crossEntropyFromData(target_data,input_data,bins):
  start=min(target_data+input_data)
  end=max(target_data+input_data)
  target_dist=getProb(createBin(target_data,start,end,bins))
  input_dist=getProb(createBin(input_data,start,end,bins))
  entropy=crossEntropy(target_dist,input_dist)
  return entropy

def reformData(data1,data2):
  data=data1+data2
  random.shuffle(data)
  ret1=data[:len(data1)]
  ret2=data[len(data1):]
  return ret1,ret2

def getP(reverse_sorted_dist,value):
  if value >reverse_sorted_dist[0]:
    return 0
  for i in range(1,len(reverse_sorted_dist)):
    if value<=reverse_sorted_dist[i-1] and value >=reverse_sorted_dist[i]:
      return float(i)/float(len(reverse_sorted_dist))
  return 1

def getSimilarityP(data1,data2,permutation):
  cross_entropy=crossEntropyFromData(data1,data2,10)
  target_entropy=crossEntropyFromData(data1,data1,10)
  diff=abs(target_entropy-cross_entropy)
  permutated_diff=[]
  for i in range(0,permutation):
    permutated_data1,permutated_data2=reformData(data1,data2)
    permutated_diff.append(abs(crossEntropyFromData(permutated_data1,permutated_data2,20)-crossEntropyFromData(permutated_data1,permutated_data1,20)))
  permutated_diff=sorted(permutated_diff,reverse=True)
  p=getP(permutated_diff,diff)
  return p

def plotGroupDensity(data,pngfile):
  datas=[]
  keys=[]
  fig = plt.figure()
  for key in data:
    datas.append(data[key])
    keys.append(key)
    density = gaussian_kde(data[key])
    xs=np.linspace(min(data[key]),max(data[key]),1000)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    ys=density(xs)
    mean=np.mean(data[key])
    stdev=np.std(data[key])
    plt.plot(xs,ys,label=key+": mean="+str("{:5.4f}".format(mean))+", "+str("{:5.4f}".format(stdev)))
  text=""
  for i in range(0,len(datas)):
    for j in range(i+1,len(datas)):
      p=getSimilarityP(datas[i],datas[j],10000)
      text+=keys[i]+"-"+keys[j]+",p="+str("{:5.4f}".format(p))+"\n"
      print keys[i]+"-"+keys[j]+",p="+str("{:5.4f}".format(p)),
  #    text+=keys[i]+"-"+keys[j]+",p="+str("{:5.4f}".format(isSameDist(datas[i],datas[j])))+"\n"
  print ""
  fig.suptitle(text)
  plt.legend(loc="best")
  plt.savefig(pngfile)
  plt.close()

def statOneKernelMaxvalue(one_kernel,kernel_id):
  max_values=[]
  types=dict()
  #types["f"]=[]
  #types["r"]=[]
  #types["b"]=[]
  for line in one_kernel:
    max_values.append(float(line[2]))
    if line[3] not in types:
      types[line[3]]=[]
    types[line[3]].append(float(line[2]))
  plotDensity(max_values,kernel_id+".png") 
  plotGroupDensity(types,kernel_id+"_detail.png")


def drawMotifLogo(one_kernel,maxvalue,kernel_id):
  f=open("tmp.fa","w")
  for line in one_kernel:
    max_value=float(line[2])
    if max_value<maxvalue:
      continue
    f.write(">\n"+line[1]+"\n")
  f.close()
  os.system("weblogo --format png_print --sequence-type dna --fout \""+kernel_id+"_weblogo.png\" -f tmp.fa")
 
def drawMotifLogoPctg(one_kernel,pctg,kernel_id):
  max_values=[]
  for line in one_kernel:
    max_values.append(float(line[2]))
  max_values=sorted(max_values,reverse=True)
  rank=int(len(max_values)*pctg) 
  threshold=max_values[rank]
  drawMotifLogo(one_kernel,threshold,kernel_id)  



def statKernels(kernelpath,total_seq):
  f=open(kernelpath)
  max_values=[]
  kernels=dict()
  for line in f:
    line=line.rstrip("\n").split("\t")
    if line[1] == "seq":
      continue
    max_values.append(float(line[2]))
    if line[0] not in kernels:
      kernels[line[0]]=[]
    kernels[line[0]].append(line)
  f.close()
  plotDensity(max_values, "max_values_density_add_base_change.png")
  for kernel in kernels:
    statOneKernelMaxvalue(kernels[kernel],kernel)
    #drawMotifLogo(kernels[kernel],0.3,kernel)
  # drawMotifLogoPctg(kernels[kernel],0.05,kernel)

  



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
  getActivatedMotifFromEachKernel(model,all_in_x,all_in_err_type,all_in_predict,kernel_length,all_in_x_marked,all_in_base_change)  
  statKernels("kernels_add_input_seq_for_0.95depth_add_base_change.txt",len(all_in_x))
  







  
  
   


    
  
  
