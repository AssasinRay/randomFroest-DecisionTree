import sys
from LoadData import *
from DecisionTreeFunctions import *
from RandomForestFunctions import *

train = sys.argv[1]
test  = sys.argv[2]

train_data, train_features = Load_Data(str(train))

test_data , test_features  = Load_Data(str(test))

decision_forest = RandomForest(train_data, train_features)
classify_res  = RandomForest_classify(decision_forest, test_data)

matrix = Print_Confusion_Matrix(train_data,test_data, classify_res)
