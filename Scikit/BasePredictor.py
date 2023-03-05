from sklearn.linear_model import LinearRegression
import pandas as pd

class BasePredictor:
    label = 'Yds'
    yTrain = None
    xTrain = None
    yTest = None
    xTest = None
    model = None
    def __init__(self, player):
        self.name = player.name
        self.team = player.team
        self.ID = player.ID
        self.testData = player.testData
        self.trainData = player.trainData
        self.allData = player.allData

    def createPredictor(self):
        return

    def train(self):
        return

    def test(self):
        return

    def predict(self):
        return

    def printPredictions(self):
        return
    