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



class Player: 
   label = 'Yds'
   trainingData = None
   testData = None
   def __init__(self, train, test):
      Player.trainingData = pd.read_csv(train,index_col=0)
      Player.testData = pd.read_csv(test,index_col=0)
 
   def train(train):
      trainingData = pd.read_csv(train,index_col=0)
      Player.y_train = trainingData[Player.label] # values to predict
      Player.x_train = trainingData.drop(columns=[Player.label]) # features

   def test(test):
      testData = pd.read_csv(test,index_col=0)
      Player.y_test = testData[Player.label]
      Player.x_test = testData.drop(columns=[Player.label]) # features

   def predict(features):
      # trainingData = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ6B026KVaZ2LrEZOq_eVe4mJN5kvvb48qitdightknV8DUnypVyfnPBjTvfpcGgds5ny_rSlR_NS4/pub?gid=1129893861&single=true&output=csv',index_col=0)
      Player.y_train = Player.trainingData[Player.label] # values to predict
      Player.x_train = Player.trainingData.drop(columns=[Player.label]) # features

      # testData = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ6B026KVaZ2LrEZOq_eVe4mJN5kvvb48qitdightknV8DUnypVyfnPBjTvfpcGgds5ny_rSlR_NS4/pub?gid=1876485471&single=true&output=csv',index_col=0)
      # Player.y_test = testData[Player.label]
      Player.x_test = Player.testData.drop(columns=[Player.label])

      model = LinearRegression()
      model.fit(Player.x_train, Player.y_train)
      Player.predictions = model.predict(Player.x_test)


   def print():
      print(Player.predictions)

p1 = Player('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ6B026KVaZ2LrEZOq_eVe4mJN5kvvb48qitdightknV8DUnypVyfnPBjTvfpcGgds5ny_rSlR_NS4/pub?gid=1129893861&single=true&output=csv', 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ6B026KVaZ2LrEZOq_eVe4mJN5kvvb48qitdightknV8DUnypVyfnPBjTvfpcGgds5ny_rSlR_NS4/pub?gid=1876485471&single=true&output=csv')
p1.predict()
print(p1.predictions)


'''
class Player: 
   label = 'Yds'

   def __init__(self, train, test):
      self.trainingData = pd.read_csv(train, index_col=0)
      self.testData = pd.read_csv(test, index_col=0)
 
   def train(self):
      self.y_train = self.trainingData[Player.label] # values to predict
      self.x_train = self.trainingData.drop(columns=[Player.label]) # features

   def test(self):
      self.y_test = self.testData[Player.label]
      self.x_test = self.testData.drop(columns=[Player.label]) # features

   def predict(self):
      self.train()
      self.test()

      model = LinearRegression()
      model.fit(self.x_train, self.y_train)
      self.predictions = model.predict(self.x_test)

   def print_predictions(self):
      print(self.predictions)

train_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ6B026KVaZ2LrEZOq_eVe4mJN5kvvb48qitdightknV8DUnypVyfnPBjTvfpcGgds5ny_rSlR_NS4/pub?gid=1129893861&single=true&output=csv'
test_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ6B026KVaZ2LrEZOq_eVe4mJN5kvvb48qitdightknV8DUnypVyfnPBjTvfpcGgds5ny_rSlR_NS4/pubhtml?gid=1876485471&single=true'

p1 = Player(train_url, test_url)
p1.predict()
p1.print_predictions()
'''