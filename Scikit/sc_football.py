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
from Player import Player
from Game import Game
from FetchFootballData import FetchFootballData
from SKlearnPredictor import SKlearnPredictor
from AutoPredictor import AutoPredictor

def runPredictions(predictor,player):
    print()
    print(predictor.__class__.__name__ + " model")
    predictor.train()
    predictor.test()
    predictor.predict()
    predictor.printPredictions()
    predictor.featureImportance()
Player
Players = FetchFootballData.fetch('https://docs.google.com/spreadsheets/d/e/2PACX-1vQQ6B026KVaZ2LrEZOq_eVe4mJN5kvvb48qitdightknV8DUnypVyfnPBjTvfpcGgds5ny_rSlR_NS4/pub?gid=1874466671&single=true&output=csv')
chubb = Players[0]
auto = AutoPredictor(chubb)
runPredictions(auto, chubb)
'''
for player in Players:
    player.print_player()
    Auto = AutoPredictor(player)
    SK = SKlearnPredictor(player)
    runPredictions(SK, player)
    runPredictions(Auto, player)
    print()
    '''