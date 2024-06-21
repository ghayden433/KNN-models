import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
"""
Author: Hayden Gillen
Date: 06/19/2024
Purpose: poke around with a simple machine learning model (K Nearest Neighbors)
DataSet: Magic Gamma Telescope https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
"""


# label the columns and read in data as a dataframe
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
print(df.head())

# randomize then split data into a training set and test set (80% and 20%)
df = df.sample(frac = 1)
train = df[1:int(0.8*len(df))]
test = df[int(0.8*len(df)):]





# determines euclidican distance from one row to another row
def distance(trainRow, testRow):
    total = 0
    for i in range(len(testRow) - 1):
        total += (trainRow.loc[cols[i]] - testRow.loc[cols[i]]) * (trainRow.loc[cols[i]] - testRow.loc[cols[i]])
    return math.sqrt(total)
        


# determines the class of testRow based on the k nearest values
def classification(trainSet, testRow, k):
    #tuple to hold the closest k rows distance and class
    closest = [(2100000000, "")] * k
    
    # check the test row agains all rows in the training set and store the k closest rows
    for i in range(len(trainSet) - 1):
        dist = distance(trainSet.iloc[i], testRow)
        # if the distance is one of the k least items, remove the max and add dist
        if dist < max(closest)[0]:
            closest.remove(max(closest))
            closest.append((dist, trainSet.iloc[i][-1]));
   
    #determine the class of testRow based on the most common class of the closest k elements
    g = 0
    h = 0
    for item in closest:
        if item[1] == 'g':
            g += 1
        else:
            h += 1
    
    if h > g:
        return "h"
    else:
        return "g"
    
    
    
# returns the results of how accurate the model is in terms of percentage of correct classifications
def KNN(trainSet, testSet, k):
    correct = 0
    total = 0
    for i in range(1, len(testSet)):
        classify = classification(trainSet, testSet.iloc[i], k)
        if classify == testSet.iloc[i].loc["class"]:
            correct += 1
        total += 1;
        print(total, (classify, testSet.iloc[i].loc["class"]), str((correct / total) * 100) + "%")
    
    return correct / total


KNN(train, test, 5)