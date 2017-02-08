
from numpy import genfromtxt
import gzip, six.moves.cPickle as cPickle
from glob import glob
import numpy as np
import pandas as pd
import csv
import math
#Data = dir_to_dataset("C:\Users\Divya Chopra\Downloads\breast-cancer-wisconsin.csv")
# Data and labels are read

DataTraining = np.genfromtxt('C:\h\kforesttraining.csv', delimiter=',', dtype='f8')[1:]
DataTraining=DataTraining[:,1:28]
DataTraining = DataTraining[~np.isnan(DataTraining).any(axis=1)]
yTraining = np.genfromtxt('C:\h\kforesttraining.csv', delimiter=',', dtype=str)[1:]
yTrain=[]
yTraining=yTraining[:,0:1]
yTrain=[];
for i in range(0,len(yTraining)):
    character=yTraining[i]
    number = int(ord(character[0].strip()) - 96)
    print(number)
    yTrain.append(number)

yTrain= np.array(yTrain, dtype=float)

DataTest = np.genfromtxt('C:\h\kforesttesting.csv', delimiter=',', dtype='f8')[1:]
DataTest=DataTest[:,1:28]
DataTest = DataTest[~np.isnan(DataTest).any(axis=1)]
yTest = np.genfromtxt('C:\h\kforesttesting.csv', delimiter=',', dtype=str)[1:]
yTest=yTest[:,0:1]
yTesting=[]
for i in range(0,len(yTest)):
    character=yTest[i]
    number = int(float(ord(character[0].strip()) - 96))
    yTesting.append(number)
yTesting= np.array(yTesting, dtype=float)
percentage=0.5
index2=math.ceil(len(DataTraining)*0.5)

#y = y[~np.isnan(Data).any(axis=1)]
#np.random.shuffle(Data)
#np.random.shuffle(y)
#print("data", Data)

Data=np.concatenate((DataTraining,DataTest), axis=0)
y=np.concatenate((yTrain,yTesting), axis=0)

percentage=0.1
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

f = gzip.open('C:\h\kForestfile.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()