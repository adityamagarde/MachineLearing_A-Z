#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:18:01 2017

@author: adityamagarde
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#READING THE DATASET
dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#FINDING MISSING VALUES --> Use Imputer
    #no missing values

#ENCODING COLUMNS -- > Use LabelEncoder, OneHotEncoder
    #no columns to be encoded

#Splitting data into test and training set --> Use train_test_split
    #small dataset, no need to split here


#FITTING LINEAR REGRESSION --> Use LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(X, y)


#FITTING THE PILYNOMIAL REGRESSION --> Use PolynomialFeatures
polynomialRegressor = PolynomialFeatures(degree=3)
X_polynomial = polynomialRegressor.fit_transform(X)
polynomialRegressor.fit(X_polynomial, y)


#FITTING LINEAR REGRESSOR USING THE X_Polynomial VARIABLE
linearRegressor2 = LinearRegression()
linearRegressor2.fit(X_polynomial, y)


#VISUALIZING THE RESULTS
plt.scatter(X, y, color='red')
plt.plot(X, linearRegressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#VISUALIZING THE POLYNOMIAL REGRESSION BASICS
plt.scatter(X, y, color='red')
plt.plot(X, linearRegressor2.predict(polynomialRegressor.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#PREDICTING WITH LINEAR REGRESSION
yLinearReg = linearRegressor.predict(X);


#PREDICTING WITH POLYNOMIAL REGRESSION
yPolynomialReg = linearRegressor2.predict(polynomialRegressor.fit_transform(X))
