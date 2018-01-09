#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 21:42:36 2018

@author: adityamagarde
"""

##THIS IS FOR A SINGLE DIMESION


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING DATASET
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#FILLING THE MISSING VALUES --> Imputer
    #no missing values here
    
#ENCODING THE VARIABLES --> LabelEncoder, OneHotEncoder
    #no labels to be encoded
    
#FEATURE SCALING -->StandardScaler
    #no need of feature scaling here

#SPLITTING THE DATASET -->train_test_split
    #very small dataset, no point of splitting

#FITTING THE DECISION TREE REGRESSION TO OUR DATASET
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

#PREDICTING THE OUTPUT
y_pred = regressor.predict(6.5)

#VISUALIZING THE RESULTS
#ADDING MORE RESOLUTION
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Salary Prediction (DECISION TREE MODEL)")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()