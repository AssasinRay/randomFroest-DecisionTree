from __future__ import division
import math
import operator
from LoadData import *
from DecisionTreeFunctions import *
from RandomForestFunctions import *
from sklearn.metrics import *
import numpy as np
import sys
from types import *
train_paths = ["balance-scale.train", "led.train.new", "nursery.data.train", "poker.train"]
test_paths  = ["balance-scale.test", "led.test.new", "nursery.data.test", "poker.test"]
def extrac_dict(data_dict):
	res = []
	for row, label_feature in data_dict.iteritems():
		res.append(label_feature['label'])

	return res

def cal_accuracy(y_true, y_pred):
	res = 0.0
	correct = 0
	incorrect = 0
	total = len(y_pred)
	for i in xrange(total):
		if y_pred[i] != y_true[i]:
			incorrect+=1
		else:
			correct+=1

	print "incorrect: ",incorrect
	print "correct: ", correct
	print "total: ", total
	return float(correct/total)

def cal_sensitivity(c_m):
	k = len(c_m)
	if k == 2:
		return float(c_m[0][0] / (c_m[0][0] + c_m[0][1]))


	sumrow = np.zeros(k)
	res = []

	for i in xrange(0,k):
		for j in xrange(0,k):
			sumrow[i]+= c_m[i][j]

	#print sumrow

	for i in xrange(0,k):
		#print c_m[i][i]
		if sumrow[i] == 0 : res.append(-1)
		else :res.append(float(c_m[i][i]/sumrow[i]))

	return res


def cal_specificity(c_m):
	k = len(c_m)
	if k == 2:
		return float(c_m[1][1] / (c_m[1][0] + c_m[1][1]))


	sumcol = np.zeros(k)
	res = []
	dignolsum = 0
	for i in xrange(0,k):
		dignolsum+=c_m[i][i]


	for i in xrange(0,k):
		for j in xrange(0,k):
			sumcol[j]+= c_m[i][j]
	#print sumcol

	for i in xrange(0,k):
		if sumcol[i] - c_m[i][i] == 0 :
			res.append(-1)
		else:
			res.append(float( (dignolsum - c_m[i][i])/(sumcol[i] - c_m[i][i])))

	return res

def decision_evaluate():
	for i in xrange(4):
		print "train file is: ", train_paths[i]
		print "test  file is: ", test_paths[i]
		print "Decision tree"

		train_data, train_features = Load_Data(train_paths[i])
		test_data , test_features  = Load_Data(test_paths[i])
		decision_tree = Create_Tree(train_data, train_features)
		y_pred  = classify(decision_tree, test_data)
		y_true 	  = extrac_dict(test_data)
		
		c_m = confusion_matrix(y_true, y_pred)
		print c_m , "\n"
		acc = cal_accuracy(y_true, y_pred)
		print "Accuracy" , acc , "\n"

		sen = cal_sensitivity(c_m)
		print "Sensitivity", sen ,"\n"

		spec = cal_specificity(c_m)
		print "Specificity", spec ,"\n"

		pre = precision_score(y_true, y_pred, average=None)
		print "Precision", pre ,"\n"

		f1 = f1_score(y_true, y_pred ,average=None)  
		print "F-1 Score", f1 , "\n"

		beta05 = fbeta_score(y_true, y_pred, beta=0.5 ,average=None)  
		print "F beta  0.5", beta05 ,"\n"

		beta2  = fbeta_score(y_true, y_pred, beta=2 , average=None) 
		print "F beta  2", beta2, "\n"

		print "\n"




def RandomForest_evaluate():
	for i in xrange(4):
		print "train file is: ", train_paths[i]
		print "test  file is: ", test_paths[i]
		print "RandomForest"

		train_data, train_features = Load_Data(train_paths[i])
		test_data , test_features  = Load_Data(test_paths[i])
		decision_forest = RandomForest(train_data, train_features,50)
		y_pred  = RandomForest_classify(decision_forest, test_data)
		y_true 	  = extrac_dict(test_data)
		
		c_m = confusion_matrix(y_true, y_pred)
		print c_m , "\n"
		acc = cal_accuracy(y_true, y_pred)

		print "Accuracy" , "%.2f" % acc , "\n"

		sen = cal_sensitivity(c_m)
		if type(sen) is ListType: 
			sen2 =  ["%.2f" % v for v in sen]
		else:
			sen2 = sen
		print "Sensitivity",  sen2 ,"\n"

		spec = cal_specificity(c_m)
		if type(spec) is ListType: 
			spec2 =  ["%.2f" % v for v in spec]
		else:
			spec2 = spec
		print "Specificity", spec2 ,"\n"

		pre = precision_score(y_true, y_pred, average=None)
		if type(pre) is ListType: 
			pre2 =  ["%.2f" % v for v in pre]
		else:
			pre2 = pre
		print "Precision",  pre2 ,"\n"

		f1 = f1_score(y_true, y_pred ,average=None)  
		if type(f1) is ListType: 
			f12 = ["%.2f" % v for v in f1]
		else:
			f12 = f1
		print "F-1 Score", f12, "\n"

		beta05 = fbeta_score(y_true, y_pred, beta=0.5 ,average=None)  
		if type(beta05) is ListType: 
			beta052 =  ["%.2f" % v for v in beta05]
		else:
			beta052 = beta05
		print "F beta  0.5", beta052,"\n"

		beta2  = fbeta_score(y_true, y_pred, beta=2 , average=None) 
		if type(beta2) is ListType: 
			beta22 =  ["%.2f" % v for v in beta2]
		else:
			beta22 = beta2
		print "F beta  2", beta22, "\n"

		print "\n"


decision_evaluate()
#RandomForest_evaluate()