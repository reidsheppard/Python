from array import array
from cgi import test
from lightgbm import LGBMRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.svm import LinearSVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing

# Only works with numberical values
imputer = KNNImputer(n_neighbors= 2, weights = 'uniform')


trainingData = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRaYPRkDlXo4_PvOwE55UqyO1oEQP4spBXEPy34mtTixFdgEDmxne0LleWT8hzgqqGDdoi75LWP0DVP/pub?gid=1879927646&single=true&output=csv', index_col=0)
testData = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRaYPRkDlXo4_PvOwE55UqyO1oEQP4spBXEPy34mtTixFdgEDmxne0LleWT8hzgqqGDdoi75LWP0DVP/pub?gid=1056049758&single=true&output=csv', index_col=0 )


imputer.fit(trainingData)
transformed_testData = imputer.transform(testData)

print(trainingData)

print("transformed_testData")
print(transformed_testData)


