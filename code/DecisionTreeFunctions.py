#DecisionTreeFunctions
from __future__ import division
import math
import operator
from LoadData import *
import random

'''Gini Calculation'''

def Calculate_Gini(data_dict):
	label_dict = Get_Label_information(data_dict)
	res = 1.0
	toatl_len = len(data_dict)

	for label_val, num in label_dict.iteritems():
		res = res - pow(num/toatl_len,2)
	return res

def Calculate_Gini_feature(data_dict, feature):
	D_len = len(data_dict)
	gini_val = 0
	for val in Find_features_val(data_dict, feature):
		sub_data_dict = Find_example_feature(data_dict, feature, val)
		freq = float( len(sub_data_dict)/ D_len)
		#print "gini_val",gini_val, "feature id ", feature
		gini_val+= freq * Calculate_Gini(sub_data_dict)

	return gini_val



'''Data Selection Function'''
#find the number of each label in data_dict
def Get_Label_information(data_dict):
	label_dict = {}

	for row, label_feature in data_dict.iteritems():
		label_id = label_feature['label']
		if label_id not in label_dict.keys():
			label_dict[label_id] = 1
		else:
			label_dict[label_id]+=1

	return label_dict


def choose_best(data_dict, feature_ids):
	min_gini = 999
	res_id   = -1
	data_dict_copy = data_dict
	for feature in feature_ids:
		gini_val = Calculate_Gini_feature(data_dict_copy, feature)
		if gini_val <= min_gini:
			res_id 		= feature
			min_gini 	= gini_val

	return res_id

#Find feature categrioal number  return a set
def Find_features_val(data_dict, feature):
	feature_val = set()
	for row, label_feature in data_dict.iteritems():
		value = label_feature['feature'][feature]
		#print value
		feature_val.add(value)

	return feature_val


#get a sub data dict by feature == val 
def Find_example_feature(data_dict, feature, val ):
	sub_data_dict = {}
	for row, label_feature in data_dict.iteritems():
		if label_feature['feature'][feature] == val :
			sub_data_dict[row] = {}
			sub_data_dict[row]['label'] = label_feature['label']
			sub_data_dict[row]['feature'] = label_feature['feature']

	return sub_data_dict


def randomlize_feature(feature_ids):
	feature_list = list(feature_ids)
	k 			 = random.randint(1,len(feature_ids))
	subset		 =  random.sample(feature_list, k)
	return set(subset)
	
'''classifier '''
def Create_Tree(data_dict, feature_ids, mode = 'single'):
	#print feature_ids
	#print len(data_dict)
	data_dict_copy = data_dict
	#base case
	# all have same label 
	label_info = Get_Label_information(data_dict)
	label_num  = len(label_info.keys())
	if label_num == 1 :
		return str(label_info.keys()[0])

	# attributes empty  return majority vote
	elif (len(feature_ids) == 0 or data_dict == None):
		return str(max(label_info.iteritems(), key=operator.itemgetter(1))[0])

	else:
		if mode == 'single':
			best_attr = choose_best(data_dict, feature_ids)
		if mode == 'forest':
			feature_ids = randomlize_feature(feature_ids)
			#print feature_ids
			best_attr = choose_best(data_dict, feature_ids)
		tree = {best_attr:{}}
		pruned_feature_ids = set([val for val in feature_ids if val != best_attr])
		#print pruned_feature_ids
		for val in Find_features_val(data_dict_copy, best_attr):
			sub_tree = Create_Tree(Find_example_feature(data_dict_copy, best_attr, val ), pruned_feature_ids)
			tree[best_attr][val] = sub_tree

	return tree


def classify(decision_tree, test_data_dict):
	classification = []

	for row_num, label_feature in test_data_dict.iteritems():
		classification.append(Predict_each_row(decision_tree, label_feature['feature']))

	for i in xrange(len(classification)):
		classification[i] = int(classification[i])

	return classification

def Predict_each_row(tree, row):

	if isinstance(tree, basestring):
		return tree
	else:
		feature = tree.keys()[0]
		fit = row[feature]
		if fit not in tree[feature].keys():
			first_key = tree[feature].keys()[0]
			sub_tree =  tree[feature][first_key]
		else:
			sub_tree = tree[feature][fit]

		return Predict_each_row(sub_tree, row)





'''Debug'''
'''
data_dict, feature_ids = Load_Data("balance-scale.train")
test_dict, test_features = Load_Data("balance-scale.test")
label_num = Get_Label_information(data_dict)
print "label info" , label_num

print "feature_ids", feature_ids

feature_val1 = Find_features_val(data_dict, 2)
print "feature_val 2", feature_val1

sub_data_dict = Find_example_feature(data_dict, 4 , 4)
#print "pruned data", sub_data_dict

val = Calculate_Gini(sub_data_dict)
print "gini", val

b_a = choose_best(data_dict,feature_ids)
#print b_a

print "tree test"
tree = Create_Tree(data_dict, feature_ids)
#print tree
res = classify(tree, test_dict)
print len(res)

wrong =0 
for i in xrange(len(res)):
	if test_dict[i]['label'] != res[i]:
		wrong+=1

print wrong
print "acc", (len(res) - wrong)/len(res)


selftest = classify(tree, data_dict)
wrongg =0 
for i in xrange(len(selftest)):
	if data_dict[i]['label'] != selftest[i]:
		wrongg+=1

print wrongg
print "acc", (len(selftest) - wrongg)/len(selftest)

b = Print_Confusion_Matrix(test_dict, res)
print b
'''