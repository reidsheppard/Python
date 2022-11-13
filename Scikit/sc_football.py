from array import array
from cgi import test
import sklearn
from lightgbm import LGBMRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.svm import LinearSVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
 
 
trainingData = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRaYPRkDlXo4_PvOwE55UqyO1oEQP4spBXEPy34mtTixFdgEDmxne0LleWT8hzgqqGDdoi75LWP0DVP/pub?gid=1879927646&single=true&output=csv', index_col=0)
testData = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRaYPRkDlXo4_PvOwE55UqyO1oEQP4spBXEPy34mtTixFdgEDmxne0LleWT8hzgqqGDdoi75LWP0DVP/pub?gid=1056049758&single=true&output=csv', index_col=0)
 
# What it uses to fill in missing values
imputer = KNNImputer(n_neighbors=4)
 
label = 'Yds'
y_train = trainingData[label] # values to predict
x_train = trainingData.drop(columns=[label]) # features
 
y_test = testData[label]
x_test = testData.drop(columns=[label]) # features
 
 
model = LinearRegression()
model.fit(x_train, y_train)
 
predictions = model.predict(x_test)
 
 
i = 1
print("Game      Yds predicted")
for x in predictions:
   print(i, "\t",x)
   i+=1
 
 
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))
 
 
 
# training the data
# imputer.fit(trainingData)
# transformed_testData = imputer.transform(testData)
 
