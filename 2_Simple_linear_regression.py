#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:19:33 2017

@author: adityamagarde
"""

#SIMPLE LINEAR REGRESSION

#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#IMPORTING DATASET
dataset = pd.read_csv('Dataset.csv')


#SPLITTING X AND Y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#SPLITTING DATA INTO TEST AND TRAINING SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


#FITTING SIMPLE LINEAR REGRESSION TO THE TRAINING SET
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#PREDICITING TEST SET RESULTS
y_pred = regressor.predict(X_test)


#VISUALIZING THE TRAINING SET RESULTS
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


#VISUALIZING THE TEST SET RESULTS
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
