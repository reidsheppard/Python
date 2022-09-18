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

df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRaYPRkDlXo4_PvOwE55UqyO1oEQP4spBXEPy34mtTixFdgEDmxne0LleWT8hzgqqGDdoi75LWP0DVP/pub?gid=1767393397&single=true&output=csv', index_col=0)

print(df.shape)
df.head()