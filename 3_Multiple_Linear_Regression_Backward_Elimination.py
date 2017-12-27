#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:22:14 2017

@author: adityamagarde
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#IMPORTING DATASET
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#ENCODING VARIABLES
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()


#AVOIDING DUMMY VARIABLE TRAP
X = X[:, 1:]

#SPLITTING DATASET INTO TEST AND TRAIN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 0)



#FITTING MULTIPLE LINEAR REGRESSION MODEL 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#PREDICTING THE RESULTS FROM TEST SET
y_pred = regressor.predict(X_test)


#BUILDING THE OPTIMAL MODEL USING BACKWARD ELIMINATION
import statsmodels.formula.api as sm
X = np.append(np.ones((50, 1)), X, axis=1)

X_opt = X[:,[0, 1, 2, 3, 4, 5]]
X_optAD = X
regressor_OLS = sm.OLS(endog = y, exog = X_optAD).fit()





