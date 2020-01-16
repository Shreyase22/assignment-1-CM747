
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn import tree

np.random.seed(42)


# Reading Data
data = pd.read_excel("./datasets/Real estate valuation data set.xlsx")
data = data.drop(["No"], axis=1)


# 70:30 train test split
train_test_split = int(0.7*data.shape[0])

X = data.iloc[:train_test_split, :-1]
X_test = data.iloc[train_test_split:, :-1]
y = data.iloc[:train_test_split, -1]
y_test = data.iloc[train_test_split:, -1]


maxdepth = 4

# Building Decesion Tree based on my model
criteria = 'information_gain'
mytree = DecisionTree(criterion=criteria, max_depth=maxdepth) #Split based on Inf. Gain
mytree.fit(X, y)
mytree.plot()

print("My Model")
y_hat = mytree.predict(X)
print("Train Scores:")
print('\tRMSE: ', rmse(y_hat, y))
print('\tMAE: ', mae(y_hat, y))

y_test_hat = mytree.predict(X_test)
print("Test Scores:")
print('\tRMSE: ', rmse(y_test_hat, y_test))
print('\tMAE: ', mae(y_test_hat, y_test))

###################################################################################

# Building Decesion Tree based on sklearn
print("Sklearn Model")
clf = tree.DecisionTreeRegressor(max_depth=maxdepth)
clf = clf.fit(X,y)
y_test_hat = pd.Series(clf.predict(X_test))
print("Test Scores:")
print('\tRMSE: ', rmse(y_test_hat, y_test))
print('\tMAE: ', mae(y_test_hat, y_test))