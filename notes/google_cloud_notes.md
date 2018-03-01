# Google Cloud Notes
Notes from the DSS Google Cloud March Madness Challenge

## Understanding the Data
Stage 1 predicting 2014, 2015, 2016, 2017.
2014, 2015, 2016 can be training data, 2017 is testing data
2018 is final round testing data. 
Predict based on probability of each team winning and calculating 
avg(-ln(prob)) 0.693 is just average

## Predictors for Data
IMPORTANT POSITIVE CORRELATED FEATURES
-Team's seed in the NCAA tournament ***Important because easier schedule (like warriors #1 seed)
-win-loss record (Very important too, shows strength of team)
-point differential *** (Hidden data not mentioned)
-National rankings, pomeroy, sagarin, espn, rpi. (very good ranking idea)
-ensembles of national rankings
-prediction from offsmakers.
-offensive and deffensive efficiency (per 100 possessions rather than per game?)
-Statistical measures of efficiency (free throw percentage, 3-point percentage, ratio of assists to turnovers, rebounding margin, effective shooting percentage, etc.)
-
OTHER POSITIVE CORRELATED FEATURES
-star players?
-injury to star players?
-team depth
-strength of opponents (not really fair, should be combined with point differential to get how good the team is)
-Players rating???
-recent performace/streak?
-home-court advantage
-three point percentage/team variance????
-team work
-best starting 5 rating

PREDICTIVE/TRENDS
-past year results...
-lost key players from past years...
-coaches record

IMPORTANT NEGATIVE FEATURES
-injuries!!!! especially to star players
https://docs.google.com/document/d/1DNrSxiIGCrbFXKW76CSnKJcr5JkorQCr7t8ufW6RTwM/edit



##Extra Notes
Google cloud has speech, vision, and NLP APIs, and ML models
Google cloud query (Extra stuff, takes too much work and little return)

