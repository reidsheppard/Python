from Game import Game
from sklearn.linear_model import LinearRegression
import pandas as pd
class Player:
    label = 'Yds'
    yTrain = None
    xTrain = None
    yTest = None
    xTest = None
    model = None
    def __init__(self, name, team, ID, testData, trainData, allData):
        self.name = name
        self.team = team
        self.ID = ID
        self.testData = testData
        self.trainData = trainData
        self.allData = allData
    
    def print_player(self):
        print(f"Name: {self.name}")
        print(f"Team: {self.team}")
        print(f"ID: {self.ID}")
        print(f"Test data: {self.testData}")
        print(f"Train data: {self.trainData}")
        print(f"All data: {self.allData}")

    def train(self):
      self.trainData = pd.read_csv(self.trainData,index_col=0)
      self.yTrain = self.trainData[self.label] # values to predict
      self.xTrain = self.trainData.drop(columns=[self.label]) # features

    def test(self):
      self.testData = pd.read_csv(self.testData,index_col=0)
      selfyTest = self.testData[self.label]
      self.xTest = self.testData.drop(columns=[self.label]) # features

    def predict(self):
      # trainingData = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ6B026KVaZ2LrEZOq_eVe4mJN5kvvb48qitdightknV8DUnypVyfnPBjTvfpcGgds5ny_rSlR_NS4/pub?gid=1129893861&single=true&output=csv',index_col=0)
      self.yTrain = self.trainData[self.label] # values to predict
      self.xTrain = self.trainData.drop(columns=[self.label]) # features

      # testData = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ6B026KVaZ2LrEZOq_eVe4mJN5kvvb48qitdightknV8DUnypVyfnPBjTvfpcGgds5ny_rSlR_NS4/pub?gid=1876485471&single=true&output=csv',index_col=0)
      # Player.y_test = testData[Player.label]
      self.xTest = self.testData.drop(columns=[self.label])

      self.model = LinearRegression()
      self.model.fit(self.xTrain, self.yTrain)
      Player.predictions = self.model.predict(self.xTest)
    def printPredictions(self):
        print(f"Predictions: {Player.predictions}")