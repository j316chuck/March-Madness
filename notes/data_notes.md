## DATA INFO
There is a lot of data for this challenge and so this document serves to clarifythe important features of the data that is useful

## Teams.csv
-identifies the team ID, team name, and amount of seasons played

## Seasons.csv
-DayZero - tells you the date corresponding to daynum=0 during that season. All game dates have been aligned upon a common scale so that the championship game of the final tournament is on daynum=154. Working backward, the national semifinals are always on daynum=152, the "play-in" games are on days 134/135, Selection Sunday is on day 132, and so on. All game data includes the day number in order to make it easier to perform date calculations. If you really want to know the exact date a game was played on, you can combine the game's "daynum" with the season's "dayzero". For instance, since day zero during the 2011-2012 season was 10/31/2011, if we know that the earliest regular season games that year were played on daynum=7, they were therefore played on 11/07/2011.

## NCAATourneySeeds.csv 
-Very Important because Seeds means easier playoff route and greater likelihood of winning championship
-Season, seed, and TEAMID (based on teams.csv)

## RegularSeasonCompactResults.csv
-Season, Day, Winning Team, Winning Score, Losing Team, Losing Score

## NCAATourneyCompactResults
-Results of NCAA in a same way as Regular Season Compact 

## Team Box Scores
- WFGM - field goals made (by the winning team)
WFGA - field goals attempted (by the winning team)
WFGM3 - three pointers made (by the winning team)
WFGA3 - three pointers attempted (by the winning team)
WFTM - free throws made (by the winning team)
WFTA - free throws attempted (by the winning team)
WOR - offensive rebounds (pulled by the winning team)
WDR - defensive rebounds (pulled by the winning team)
WAst - assists (by the winning team)
WTO - turnovers committed (by the winning team)
WStl - steals (accomplished by the winning team)
WBlk - blocks (accomplished by the winning team)
WPF - personal fouls committed (by the winning team)

## NCAATourneyDetailedResults and RegularSeasonDetaileResults
-contains detailed statistics. of aforementioned

## MasseyOrdinals
-contains ranking system, #1 #2 ... #N of rankings
-could analyze trend of rankings

## Play By Play 
-includes all the plays per game (probably not very useful)
-events or players (list names of all players)

## Team Coaches
-team coaches.csv (lists coaches and id number)

## NCAA Tourney Slots
-Figures out how teams are played against each other





