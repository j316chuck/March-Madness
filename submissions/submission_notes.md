## Notes on Submission
The file you submit will depend on whether the competition is in stage 1 (historical model building) or stage 2 (the 2018 tournament). Sample submission files will be provided for both stages. The format is a list of every possible matchup between the tournament teams. Since team1 vs. team2 is the same as team2 vs. team1, we only include the game pairs where team1 has the lower team id. For example, in a tournament of 68 teams (64 + 4 play-in teams), you will predict (68*67)/2  = 2278 matchups. 

Each game has a unique id created by concatenating the season in which the game was played, the team1 id, and the team2 id. For example, "2013_1104_1129" indicates team 1104 played team 1129 in the year 2013. You must predict the probability that the team with the lower id beats the team with the higher id.

The resulting submission format looks like the following, where "pred" represents the predicted probability that the first team will win

id,pred
2013_1103_1107,0.5
2013_1103_1112,0.5
2013_1103_1125,0.5
