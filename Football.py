#these are comments

from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('https://docs.google.com/spreadsheets/d/e/2PACX-1vRaYPRkDlXo4_PvOwE55UqyO1oEQP4spBXEPy34mtTixFdgEDmxne0LleWT8hzgqqGDdoi75LWP0DVP/pub?gid=1767393397&single=true&output=csv')
train_data.head()

label = 'Yds'
print("Summary of class variable: \n", train_data[label].describe())

save_path = 'russhingModel-predictClass'  # specifies folder to store trained models
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)

test_data = TabularDataset('https://docs.google.com/spreadsheets/d/e/2PACX-1vRaYPRkDlXo4_PvOwE55UqyO1oEQP4spBXEPy34mtTixFdgEDmxne0LleWT8hzgqqGDdoi75LWP0DVP/pub?gid=1489051496&single=true&output=csv')
y_test = test_data[label]  # values to predict
test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
test_data_nolab.head()

predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data_nolab)
print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

predictor.leaderboard(test_data, silent=True)