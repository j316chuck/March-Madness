import pandas as pd
import numpy as np
import sys
import math
pd.options.mode.chained_assignment = None  # default='warn'

input_file = None
df = None 
submission_df = None
predictions = {}
test_years = [] 

def predict(row, year):
    mn = min(row['WTeamID'], row['LTeamID'])
    mx = max(row['WTeamID'], row['LTeamID'])
    return predictions[(year, mn, mx)]

def logloss(row):
    return (row.Result * math.log(row.Prediction) + (1.0 - row.Result) * math.log(1.0 - row.Prediction))

def evaluate(year):
    ncaa_df = df[(df['Season'] == year) & (df.DayNum >= 136)]
    assert(len(ncaa_df.index) == 63)
    ncaa_df['Result'] = ncaa_df.apply(lambda x : int(x.WTeamID < x.LTeamID), axis = 1)
    ncaa_df['Prediction'] = ncaa_df.apply(predict, args = (year,), axis = 1)
    ncaa_df['LogLoss'] = ncaa_df.apply(logloss, axis = 1)
    return -(ncaa_df['LogLoss'].sum() / len(ncaa_df.index)), ncaa_df

def test():
    log_losses = []
    for year in test_years:
        log_loss, ncaa_df = evaluate(year)
        print "Log loss of year {0} is {1}".format(year, log_loss)
        log_losses.append(log_loss)
    average = sum(log_losses) / len(log_losses)
    print "Average log loss across the years {1} - {2} is {0}".format(average, test_years[0], test_years[-1])
    return average, log_losses

def main(): 
    global input_file, df, test_years, submission_df, predictions
    input_file = sys.argv[1]
    df = pd.read_csv('../input/NCAATourneyCompactResults.csv')
    st, en = int(sys.argv[2]), int(sys.argv[3])
    test_years = range(st, en + 1)
    submission_df = pd.read_csv(input_file)
    predictions = {}
    for index, row in submission_df.iterrows():
	year, sid, eid = map(int, row['id'].split('_'))
	predictions[(year, sid, eid)] = row['pred']	
    test()
 
if __name__ == "__main__":
    main()




