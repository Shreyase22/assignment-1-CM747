import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read IRIS data set
# ...
# 

# Fetching Dataset
f =  open("datasets/iris.data","rb")
iris_data = repr(f.read())[2:-1].strip("\\n").split("\\n")

# Pre-processing Data
np.random.shuffle(iris_data)

X_data = {"sepal length": list(), "sepal width": list(), "petal length":list(), "petal width": list()}
y_data = list()
for i in range(len(iris_data)):
    row = iris_data[i].strip(" ").split(",")
    X_data["sepal length"].append(float(row[0]))
    X_data["sepal width"].append(float(row[1]))
    X_data["petal length"].append(float(row[2]))
    X_data["petal width"].append(float(row[3]))
    y_data.append(row[4])
X_data = pd.DataFrame(data=X_data)
y_data = pd.Series(data=y_data, dtype="category")


# Defining Train Test Split
train_test_split = int(0.7*len(iris_data))

X = X_data.iloc[:train_test_split, :]
X_test = X_data.iloc[train_test_split:, :]
y = y_data.iloc[:train_test_split]
y_test = y_data.iloc[train_test_split:]

# Training and Testing
for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria, max_depth=3)
    # Build Decision Tree
    tree.fit(X, y)
    #Predict
    y_hat = tree.predict(X)
    y_test_hat = tree.predict(X_test)
    tree.plot()
    print('Criteria :', criteria)
    print('Train Accuracy: ', accuracy(y_hat, y))
    print('Test Accuracy: ', accuracy(y_test_hat, y_test))
    # Precesion and Recall for each class
    for cls in y.unique():
        print("Class =",cls)
        print('Precision: ', precision(y_test_hat, y_test, cls))
        print('Recall: ', recall(y_test_hat, y_test, cls))


####################################################################################

# 5 fold cross-validation
acc = 0
for i in range(5):
    a = int((i/5)*len(iris_data))
    b = int(((i+1)/5)*len(iris_data))
    X = pd.concat([X_data.iloc[:a, :], X_data.iloc[b:,:]], ignore_index=True)
    X_test = X_data.iloc[a:b, :]
    y = pd.concat([y_data.iloc[:a], y_data.iloc[b:]], ignore_index=True)
    y_test = y_data.iloc[a:b]
    tree = DecisionTree(criterion="information_gain", max_depth=3)
    tree.fit(X, y)
    y_test_hat = tree.predict(X_test)
    acc += accuracy(y_test_hat, y_test)

print("5 fold cross-validation average accuracy:", acc/5)


####################################################################################

# Finding Optimal depth using nested cross-validation

valaccuracyarray = list()               # list used to plot validation accuracy

no_of_outer_folds = 5                   # using 5 fold cross-validation
no_of_inner_folds = 7                   # no. of folds in nested cross-validaton

acc = -1
optimaldepth = -1
for i in range(no_of_outer_folds):
    a = int((i/no_of_outer_folds)*len(iris_data))
    b = int(((i+1)/no_of_outer_folds)*len(iris_data))
    X = pd.concat([X_data.iloc[:a, :], X_data.iloc[b:,:]], ignore_index=True)
    X_test = X_data.iloc[a:b, :]
    y = pd.concat([y_data.iloc[:a], y_data.iloc[b:]], ignore_index=True)
    y_test = y_data.iloc[a:b]
    hepler_var = {"accuracy": -1, "depth":-1}
    val_accuracy = list()
    for depth in range(10):
        temp = 0
        for j in range(no_of_inner_folds):
            a = int((j/no_of_inner_folds)*X.shape[0])
            b = int(((j+1)/no_of_inner_folds)*X.shape[0])
            X_val = X.iloc[a:b, :]
            X_train = pd.concat([X.iloc[:a, :], X.iloc[b:, :]],ignore_index=True)
            y_val = y.iloc[a:b]
            y_train = pd.concat([y.iloc[:a], y.iloc[b:]],ignore_index=True)
            tree = DecisionTree(criterion="information_gain", max_depth=depth)
            tree.fit(X_train, y_train)
            y_val_hat = tree.predict(X_val)
            temp += accuracy(y_val_hat, y_val)
        temp = temp/no_of_inner_folds
        if(hepler_var["accuracy"]==-1):
            hepler_var["accuracy"] = temp
            hepler_var["depth"] = depth
        else:
            if(temp>hepler_var["accuracy"]):
                hepler_var["accuracy"] = temp
                hepler_var["depth"] = depth
        val_accuracy.append(temp)
    valaccuracyarray.append(val_accuracy)
    tree = DecisionTree(criterion="information_gain", max_depth=hepler_var["depth"])
    tree.fit(X,y)
    y_test_hat = tree.predict(X_test)
    accur = accuracy(y_test_hat, y_test)
    if(accur>acc):
        acc = accur
        optimaldepth = hepler_var["depth"]
    print("Accuracy:",accur, "depth:", hepler_var["depth"])



print("Optimal Depth:", optimaldepth)               # Showing Optimal Depth for best results


####################################################################################

# Plot results of 5 folds

deptharray = [[i for i in range(10)] for j in range(no_of_outer_folds)]

fig = plt.figure()
ax = plt.subplot()
for i in range(no_of_outer_folds):
    ax.plot(deptharray[i], valaccuracyarray[i])
ax.set_xlabel('depth')
ax.set_ylabel('accuracy')
plt.show()