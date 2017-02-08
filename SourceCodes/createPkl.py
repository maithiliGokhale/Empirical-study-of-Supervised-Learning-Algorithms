
from numpy import genfromtxt
import gzip, six.moves.cPickle as cPickle
from glob import glob
import numpy as np
import pandas as pd
import csv
import math


DataTraining = np.genfromtxt('C:\h\imagesFile.csv', delimiter=',', dtype='f8')[0:]
DataTraining = DataTraining[~np.isnan(DataTraining).any(axis=1)]
yTraining = DataTraining[:,64]
DataTraining=DataTraining[:,0:64]

print(yTraining)
DataTest = np.genfromtxt('C:\h\imagesTest.csv', delimiter=',', dtype='f8')[0:]
DataTest = DataTest[~np.isnan(DataTest).any(axis=1)]
yTest = DataTest[:,64]
DataTest=DataTest[:,0:64]

Data=np.concatenate((DataTraining,DataTest), axis=0)
y=np.concatenate((yTraining,yTest), axis=0)

percentage=0.7
index2=math.ceil(len(Data)*percentage)
DataTraining=Data[:index2]
DataTesting=Data[index2+1:]
trainVal=0.5
index3=math.ceil(len(DataTraining)*trainVal)
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

f = gzip.open('C:\h\imgfile.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()