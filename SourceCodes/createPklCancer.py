from numpy import genfromtxt
import gzip, six.moves.cPickle as cPickle
from glob import glob
import numpy as np
import pandas as pd
import csv
import math

#Data = dir_to_dataset("C:\Users\Divya Chopra\Downloads\breast-cancer-wisconsin.csv")
# Data and labels are read

Data = np.genfromtxt('C:\h\imagesFile.csv', delimiter=',', dtype='f8')[0:]
Data = Data[~np.isnan(Data).any(axis=1)]
np.random.shuffle(Data)
y = Data[:,10]
Data=Data[:,1:10]
#y = y[~np.isnan(Data).any(axis=1)]
np.random.shuffle(Data)
#np.random.shuffle(y)
#print("data", Data)
print("Y", y)



percentage=0.7
index2=math.ceil(len(Data)*0.5)
DataTraining=Data[:index2]
DataTesting=Data[index2+1:]
trainVal=0.5
index3=math.ceil(len(DataTraining)*0.5)
DataValidation=DataTraining[index3+1:]
DataTraining=DataTraining[:index3]
yTraining=y[:index3]
yValidation=y[index3+1:]
yTesting=y[index2+1:]

train_set_x = DataTraining
val_set_x = DataValidation
test_set_x = DataTesting

#print("train_set_x", train_set_x)

train_set_y =yTraining
val_set_y = yValidation
test_set_y = yTesting

#print("train_set_y", train_set_y)

# Divided dataset into 3 parts. I had 6281 images.

train_set = train_set_x,train_set_y
val_set = val_set_x,val_set_y
test_set = test_set_x,test_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('C:\h\kfile.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()