from autogluon.tabular import TabularDataset, TabularPredictor
from BasePredictor import BasePredictor
import requests
import json

class AutoPredictor(BasePredictor):
    label = 'Yds'
    yTrain = None
    xTrain = None
    yTest = None
    xTest = None
    yPred = None
    savePath = None
    testDataNoLab = None
    predictor = None
    predictions = None
    toolKit = None
    

    def train(self):
        self.trainData = TabularDataset(self.trainData)
        self.predictor = TabularPredictor(label=self.label).fit(train_data = self.trainData)
        
    def test(self):
        self.testData = TabularDataset(self.testData)
        self.yTest = self.testData[self.label]  # values to predict
        self.testDataNoLab = self.testData.drop(columns=[self.label])  # delete label column to prove we're not cheating
        self.testDataNoLab.head()

    def predict(self):
        self.predictions = self.predictor.predict(self.testData)
        print(type(self.predictions))
        

    def printPredictions(self):
        print(f"Predictions: {self.predictions}")
    
    def featureImportance(self):
        print(self.predictor.feature_importance(self.trainData))

    def getPredictions(self):
        return self.predictions
    
    def savePredictions(self):
        url = 'https://q3u75tbshfdbppxckhrfkyquxy.appsync-api.us-east-2.amazonaws.com/graphql'

        # Create a dictionary of data to send in JSON format
        data = {
            'query': '''
                mutation MyMutation($defId: Int!, $playerName: String!, $rushingYds: Int!) {
                    createPrediction(input: {def_id: $defId, player_name: $playerName, rushing_yds: $rushingYds}) {
                        id
                    }
                }
            ''',
            'variables': {
                'defId': 1,
                'playerName': 'Nick Chubb',
                'rushingYds': 100 
            }
        }


        # Convert the data dictionary to a JSON string
        payload = json.dumps(data)

        # Set the content type to JSON and add API key to headers
        headers = {
        'Content-Type': 'application/json',
        'x-api-key': 'da2-dr5p4qqekfb2jnylsqifrv6reu'
        }

        # Send a POST request with the JSON data and API key
        response = requests.post(url, data=payload, headers=headers)

        # Print the response status code and text
        print(response.status_code)
        print(response.text)        



   
'''
trainData = TabularDataset('https://docs.google.com/spreadsheets/d/e/2PACX-1vRaYPRkDlXo4_PvOwE55UqyO1oEQP4spBXEPy34mtTixFdgEDmxne0LleWT8hzgqqGDdoi75LWP0DVP/pub?gid=1767393397&single=true&output=csv')
trainData.head()

label = 'Yds'
print("Summary of class variable: \n", trainData[label].describe())

savePath = 'rushingModel-predictClass-autogluon'  # specifies folder to store trained models
predictor = TabularPredictor(label=label, path=savePath).fit(trainData)

testData = TabularDataset('https://docs.google.com/spreadsheets/d/e/2PACX-1vRaYPRkDlXo4_PvOwE55UqyO1oEQP4spBXEPy34mtTixFdgEDmxne0LleWT8hzgqqGDdoi75LWP0DVP/pub?gid=1489051496&single=true&output=csv')
yTest = testData[label]  # values to predict
testData_nolab = testData.drop(columns=[label])  # delete label column to prove we're not cheating
testData_nolab.head()

predictor = TabularPredictor.load(savePath)  # unnecessary, just demonstrates how to load previously-trained predictor from file

yPred = predictor.predict(testData_nolab)
print("Predictions:  \n", yPred)
perf = predictor.evaluate_predictions(y_true=yTest, y_pred=yPred, auxiliary_metrics=True)

predictor.leaderboard(testData, silent=True)
'''