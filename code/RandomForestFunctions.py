from __future__ import division
import math
import operator
from LoadData import *
from DecisionTreeFunctions import *
import random


def RandomForest(data_dict, feature_ids, tree_num = 50 ):

	Decision_Forest = [None]*tree_num
	for tree_id in xrange(tree_num):
		shuffled_data = bootstrap_data(data_dict)
		Decision_Forest[tree_id] = Create_Tree(shuffled_data, feature_ids , mode = 'forest')

	return Decision_Forest

def bootstrap_data(data_dict):

	total_len = len(data_dict)
	shuffle_data = {}
	for row in xrange(total_len):
		random_id = random.randint(0,total_len-1)
		shuffle_data[row] = data_dict[random_id]

	return shuffle_data

def majority_vote(all_votes_each):
	most_common = -1
	freq = {}
	for i in xrange(len(all_votes_each)):
		label  = all_votes_each[i]
		if label not in freq.keys():
			freq[label] = 1 
		else:
			freq[all_votes_each[i]]+=1

	return max(freq.iteritems(), key=operator.itemgetter(1))[0]

def RandomForest_classify(forest,data_dict):
	tree_num = len(forest)
	row_num  = len(data_dict)
	classification = [None] * row_num
	all_votes = [None] * tree_num
	for i in xrange(tree_num):
		all_votes[i] = classify(forest[i],data_dict)

	for row in xrange(row_num):
		all_votes_each = [None] * tree_num
		for tree_id in xrange(tree_num):
			all_votes_each[tree_id] = all_votes[tree_id][row]

		classification[row] = majority_vote(all_votes_each)

	return classification

'''Debug'''
'''
data_dict, feature_ids = Load_Data("balance-scale.train")
test_dict, test_features = Load_Data("balance-scale.test")

f = RandomForest(data_dict, feature_ids, tree_num =5)
print f[1] == f[0]
'''