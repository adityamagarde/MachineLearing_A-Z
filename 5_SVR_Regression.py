#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 00:22:55 2017

@author: adityamagarde
"""

#SVR


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


#IMPORTING THE DATASET ---> read_csv
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
y = y.reshape(-1, 1)


#FILLING IN THE MISSING VALUES --> Imputer
    #no missing values here in our dataset


#ENCODING THE VARIABLES -- > LabelEncoder, OneHotEncoder
    #no lables required to be encoded


#FEATURE SCALING -- > StandardScaler
scale_X = StandardScaler()
scale_y = StandardScaler()
X = scale_X.fit_transform(X)
y = scale_y.fit_transform(y)


#SPLITTING DATA INTO TEST AND TRAIN SET --> train_test_split
    #Small dataset, no need to split


#FITTING THE SVR TO THE DATASET
SVR_regressor = SVR(kernel = 'rbf')
SVR_regressor.fit(X, y)


#PREDICTING A NEW RESULT
y_pred = SVR_regressor.predict(X)


#VISUALIZING THE RESULTS
plt.scatter(X, y, color='red')
plt.plot(X, SVR_regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Level')
plt.ylabel('Salaries')
plt.show()

