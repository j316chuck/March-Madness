## March Madness Prediction Challenge - Kaggle
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

## Overview and Work Pipeline
- Preprocessing
- Feature Extraction
- Model Selection
- Ensemble
- Repeat and Win

## Structure of Repo
code - contains all the preprocessing, features, models, and ensembles we have done. Only commit and push to the masters code if your model will be good for ensembling. All other models and code should be left in your branch

input - has all the inputs

notes - useful and unuseful notes about the contest, also has images and graphs that might help for feature extraction

submissions - output files go here

## Work Flow 
Please update this README.md file continually so we know what models have already been tested. That way we won't repeat ourselves. Post the link and general description of the model you have tested below (accuracy, results, findings, and anything you think would be useful)


Model 1: Pomeroy Submission (Chuck)
Pomeroy model from Ken Pomeroy's rankings for offensive and defensive ratings

## Future Work
-https://www.kaggle.com/captcalculator/a-very-extensive-ncaa-exploratory-analysis
great visualization

- https://www.kaggle.com/c/march-machine-learning-mania-2017
(Each of them had simple models)

- http://blog.kaggle.com/2017/05/19/march-machine-learning-mania-1st-place-winners-interview-andrew-landgraf/
-Create own team efficiency ratings using regression models (calculate this) and distance travelled, (bayesian logistic regression model rstanarm package)
-used mle of creating sample distributions of other people's models really interesting to guess what other people would predict and how to beat them.
-Same predictions in both submissions except championship game in which each team given 100% chance of winning

- http://blog.kaggle.com/2016/05/10/march-machine-learning-mania-2016-winners-interview-1st-place-miguel-alomar/
-random forest and logit regression, 
-offensive and deffensive efficiency/how to calculate those
-what weight to give to strength of schedule. 
-key features to test is penalize a team who hasn't played against best teams, offensive vs defensive efficiency.35% reading forums, 15% manipulating data, 25% building models, and 25% evaluating results. (I think that's a great spread, 40% reading others 10% manipulating data, 25% building and testing models, 25% evaluating results)
-training time is small. 

- http://blog.kaggle.com/2015/04/17/predicting-march-madness-1st-place-finisher-zach-bradshaw/
(not useful)

- http://blog.kaggle.com/2014/04/21/qa-with-gregory-and-michael-1st-place-in-march-ml-mania/ 
(Ken Pomeroy's data and margin of victory model (actual spread posted in Las Vegas for each game), using logistic regression. weighted average of two probabilities
-Las Vegas line is absolutely increible **********
-1) simple model 2) specific loss function 3) las vegas model 4) margin of victory 5) logistic regression. 6) use previous games 7) ignored seed number
G-simple logistic regression seen multiple times for probabilities (or ridge) 

0) trained data on hundreds of game played each week
1) used features like Las Vegas Spread, Ken Pomeroy rating, and offensive vs defensive rating. (three most important, core features (gives you a shot))
How to calculate elo matters -- think about rankings ... and other stuff
2) Loss function -> logistic regression (e^0.17x / 1 + e^0.17x) -> really simple
3) Ensembled simple weighted average of the three features. (try two things, feature + ensemble, just feature)

Repeat with other less important features
At the end try different models

## Notes and ideas 
