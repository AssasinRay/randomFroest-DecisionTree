import sys
import re
from DecisionTreeFunctions import *

def Load_Data(DataFile):
	data = open(DataFile,"r")
	data_dict = {}
	row_num = 0
	feature_set = set()

	for line in data:
		if not line.strip():
			continue

		label_feature 					= line.split()
		data_dict[row_num] 				= {}
		data_dict[row_num]['label'] 	= int(label_feature[0])
		data_dict[row_num]['feature'] 	= {}
		for i in xrange(1,len(label_feature)):
			id_val 		= label_feature[i].split(':')
			feature_id  = int(id_val[0]) 
			feature_val = int(id_val[1])
			feature_set.add(feature_id)
			data_dict[row_num]['feature'][feature_id] = feature_val

		row_num = row_num+1

	data.close()

	return data_dict, feature_set

def Print_Confusion_Matrix(train_data , test_data, classification):
	k = len(Get_Label_information(train_data).keys())
	matrix = [[0]*k for _ in range(k)]

	for row in xrange(len(classification)):
		true_val 	=	test_data[row]['label'] 
		predict_val = 	classification[row]
		matrix[true_val-1 ][predict_val-1] += 1 

	clean_matrix  = matrix
	clean_matrix = '\n'.join(' '.join(str(cell) for cell in row) for row in matrix)
	print clean_matrix

	return clean_matrix

#Load_Data("balance-scale.train")