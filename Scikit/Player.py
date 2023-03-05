from BasePredictor import BasePredictor

class Player:
    label = 'Yds'
    yTrain = None
    xTrain = None
    yTest = None
    xTest = None
    model = None
    predictor = None
    toolKit = None
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
      self.predictor.train()

    def test(self):
      self.predictor.test()

    def predict(self):
      self.predictor.predict()

    def printPredictions(self):
      self.predictor.printPredictions()