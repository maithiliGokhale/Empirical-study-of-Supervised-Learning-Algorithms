from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import numpy as np
from numpy import genfromtxt, savetxt
import sys
import random

def print_misclassification_error():
	training_labels = genfromtxt(open('training.csv','r'), delimiter=',', usecols=64, dtype=str)[0:]
	training_data = genfromtxt(open('training.csv','r'), delimiter=',', dtype='f8')[:,:64]

	test_labels = genfromtxt(open('testing.csv','r'), delimiter=',', usecols=64, dtype=str)[0:]
	test_data = genfromtxt(open('testing.csv','r'), delimiter=',', dtype='f8')[:,:64]

	#RandomForest

	rf = RandomForestClassifier(n_estimators=100)
	rf.fit(training_data,training_labels)
	predict_rf = rf.predict(test_data)
	actual = np.array(test_labels)
	predicted_rf = np.array(predict_rf)
	error_rf = np.sum(actual!=predicted_rf)

	#SVM

	clf = svm.SVC(C=0.001,kernel="poly")
	clf.fit(training_data,training_labels)
	predict_svm = clf.predict(test_data)
	predicted_svm = np.array(predict_svm)
	error_svm = np.sum(actual!=predicted_svm)
	
	#AdaBoost
	aclf = AdaBoostClassifier(n_estimators=100)
	aclf.fit(training_data,training_labels)
	
	predict_aclf = aclf.predict(test_data)

	actual = np.array(test_labels)
	predicted_aclf = np.array(predict_aclf)

	error_aclf = np.sum(actual!=predicted_aclf)
	
	misclassification_error_rf = error_rf/(len(test_data)*1.0)*100
	misclassification_error_svm = error_svm/(len(test_data)*1.0)*100
	misclassification_error_aclf = error_aclf/(len(test_data)*1.0)*100
	print "Misclassification Error using Random Forest : %f" % misclassification_error_rf
	print "Misclassification Error using SVM : %f" % misclassification_error_svm
	print "Misclassification Error using AdaBoost : %f" % misclassification_error_aclf


if __name__ == '__main__':
    print_misclassification_error()














