
import pandas as pd
import numpy as np
import time
from tree.base import DecisionTree
from metrics import *
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

def createFakeData(N,P,case):
    if(case==1):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
    elif(case==2):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(P, size = N), dtype="category")
    elif(case==3):
        X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randint(P, size = N), dtype="category")
    else:
        X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randn(N))

    return X, y


def plotTimings(timeData):
    df = pd.DataFrame(data=timeData)
    heatmap1_data = pd.pivot_table(df, values='time', index=['N'], columns='P')
    sns.heatmap(heatmap1_data, cmap="YlGnBu")
    plt.show()


def analyseTime(case):
    assert(1<=case<=4)
    fitTimes = {'N':list(), 'P':list(), 'time':list()}
    predictTimes = {'N':list(), 'P':list(), 'time':list()}
    for N in range(40,50):
        for P in range(2,10):
            print("Running with N",N,"and P",P)
            X, y = createFakeData(N,P,case)
            tree = DecisionTree(criterion="information_gain", max_depth=3)
            
            startTime = time.time()
            tree.fit(X,y)
            endTime = time.time()
            fitTimes['N'].append(N)
            fitTimes['P'].append(P)
            fitTimes['time'].append(endTime - startTime)
            
            startTime = time.time()
            y_hat = tree.predict(X)
            endTime = time.time()
            predictTimes['N'].append(N)
            predictTimes['P'].append(P)
            predictTimes['time'].append(endTime - startTime)
    
    plotTimings(fitTimes)
    plotTimings(predictTimes)


analyseTime(4)
analyseTime(3)
analyseTime(2)
analyseTime(1)