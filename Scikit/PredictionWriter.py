from sklearn.linear_model import LinearRegression
import pandas as pd
from Player import Player

class PredictionWriter:
    predictions = None
    def __init__(self, predictor):
        self.predictions = predictor.predictions