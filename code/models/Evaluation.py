


#######################################################################################################################
# Model implemented off of https://github.com/adeshpande3/March-Madness-2017/blob/master/March%20Madness%202017.ipynb #
#######################################################################################################################
import cPickle as pickle
import sklearn
import pandas as pd
import numpy as np
import collections
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import tree
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from sklearn.cross_validation import cross_val_score
from keras.utils import np_utils
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
from sklearn.ensemble import GradientBoostingRegressor
import math
import csv
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import urllib
from sklearn.svm import LinearSVC
from utils import *
from sklearn import metrics
from sklearn.metrics import make_scorer
from Features import createSeasonDict
import os.path

#reading input
data_dir = '../../input/'
reg_season_compact_pd = pd.read_csv(data_dir + 'RegularSeasonCompactResults.csv')
reg_season_detailed_pd = pd.read_csv(data_dir + 'RegularSeasonDetailedResults.csv')
seasons_pd = pd.read_csv(data_dir + 'Seasons.csv')
teams_pd = pd.read_csv(data_dir + 'Teams.csv')
teamList = teams_pd['TeamName'].tolist()
tourney_compact_pd = pd.read_csv(data_dir + 'NCAATourneyCompactResults.csv')
tourney_detailed_pd = pd.read_csv(data_dir + 'NCAATourneyDetailedResults.csv')
tourney_seeds_pd = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')
conference_pd = pd.read_csv(data_dir + 'Conference.csv')
tourney_results_pd = pd.read_csv(data_dir + 'TourneyResults.csv')
NCAAChampionsList = tourney_results_pd['NCAA Champion'].tolist()

#loading feature matrix
#change model and test year and input data too get a different cv score and a different game result
start = int(sys.argv[1]) 
end = int(sys.argv[2]) + 1
test_year = range(start, end)
xTrain = np.load(data_dir + 'FeatureMatrix/' + sys.argv[3])
yTrain = np.load(data_dir + 'FeatureMatrix/' + sys.argv[4])
xTest = np.load(data_dir + 'FeatureMatrix/' + sys.argv[5])
yTest = np.load(data_dir + 'FeatureMatrix/' + sys.argv[6])
output_file = "../../submissions/" + sys.argv[7]

#models to test
model = linear_model.LogisticRegression()
#model = tree.DecisionTreeClassifier()
#model = tree.DecisionTreeRegressor()
#model = linear_model.BayesianRidge()
#model = linear_model.Lasso()
#model = svm.SVC()
#model = svm.SVR()
#model = linear_model.Ridge(alpha = 0.5)
#model = AdaBoostClassifier(n_estimators=100)
#model = GradientBoostingClassifier(n_estimators=100)
#model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
#model = RandomForestClassifier(n_estimators=64)
#model = KNeighborsClassifier(n_neighbors=39)
#neuralNetwork(10)
#model = VotingClassifier(estimators=[('GBR', model1), ('BR', model2), ('KNN', model3)], voting='soft')
#model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=0.1)


#requires predict_proba support
def cross_validation_score():
    scores = cross_val_score(model, xTrain, yTrain, cv = 5, scoring = 'neg_log_loss')
    print "Log loss: {0} (+/- {1})".format(-scores.mean(), scores.std() * 2)
cross_validation_score()


model.fit(xTrain, yTrain)
sample_sub_pd = pd.read_csv(data_dir + 'sample_submission_2015.csv')


def predictGame(team_1_vector, team_2_vector, home, model):
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    diff.append(home)
    return model.predict_proba([diff])


season_dicts = {}
for year in test_year:
    print "Creating year {}".format(year)
    season_dicts[year] = createSeasonDict(year)


def createPrediction(year, season_dict):
    results = [[0 for x in range(2)] for x in range(len(sample_sub_pd.index))]
    for index, row in sample_sub_pd.iterrows():
        matchup_id = row['id']
        year = int(matchup_id[0:4])
        team1_id = int(matchup_id[5:9])
        team2_id = int(matchup_id[10:14])
        team1_vector = season_dict[int(team1_id)]
        team2_vector = season_dict[int(team2_id)]
        pred = predictGame(team1_vector, team2_vector, 0, model)
        results[index][0] = matchup_id
        results[index][1] = pred[0][1]
    return results

def createPredictionResults():
    results = []
    for year in test_year:
        result = createPrediction(year, season_dicts[year])
        results.extend(result)
    return results

results = createPredictionResults()

def toCSV():
    firstRow = [[0 for x in range(2)] for x in range(1)]
    firstRow[0][0] = 'ID'
    firstRow[0][1] = 'Pred'
    with open(output_file, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(firstRow)
        writer.writerows(results)
toCSV()


def predict(row, year):
    mn = min(row['WTeamID'], row['LTeamID'])
    mx = max(row['WTeamID'], row['LTeamID'])
    return predictions[(year, mn, mx)]

def logloss(row):
    return (row.Result * math.log(row.Prediction) + (1.0 - row.Result) * math.log(1.0 - row.Prediction))

def evaluate(year):
    ncaa_df = tourney_detailed_pd[(tourney_detailed_pd['Season'] == year) & (tourney_detailed_pd.DayNum >= 136)]
    assert(len(ncaa_df.index) == 63)
    ncaa_df['Result'] = ncaa_df.apply(lambda x : int(x.WTeamID < x.LTeamID), axis = 1)
    ncaa_df['Prediction'] = ncaa_df.apply(predict, args = (year,), axis = 1)
    ncaa_df['LogLoss'] = ncaa_df.apply(logloss, axis = 1)
    return -(ncaa_df['LogLoss'].sum() / len(ncaa_df.index))


# In[137]:
predictions = {}
def test():
    import warnings
    warnings.filterwarnings("ignore")
    log_losses = []
    for row in results:
        year, sid, eid = map(int, row[0].split('_'))
        predictions[(year, sid, eid)] = row[1]
    
    print(predictions)
    for year in test_year:
        log_loss = evaluate(year)
        log_losses.append(log_loss)
        print "Log Loss of year {} is {}".format(year, log_loss)
    print "Log Loss average over years {} - {} is {}".format(test_year[0], test_year[-1], sum(log_losses) / len(log_losses))
    return sum(log_losses) / len(log_losses)

test()

