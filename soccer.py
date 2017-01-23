# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 07:26:38 2016

@author: Jonny
"""

import sqlite3 as sq
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree


# Read sqlite query results into a pandas DataFrame
con = sq.connect("D:\code\machine learning\data sets\soccer_6\database.sqlite")
c= con.cursor()

#c.execute("DROP TABLE IF EXISTS Away_Potential")
#c.execute("DROP TABLE IF EXISTS Home_Potential")
#con.text_factory = str  #prevent encoding error
#df = pd.read_sql_query("SELECT * from Match", con)
select_script = """ 
CREATE TABLE IF NOT EXISTS stats_of_player AS SELECT * FROM Player_Stats GROUP BY player_api_id;
CREATE INDEX IF NOT EXISTS stats_index ON stats_of_player (player_api_id); 
CREATE TABLE IF NOT EXISTS Home_Potential AS SELECT m.Id ,m.home_team_goal, m.away_team_goal, (homest1.potential + homest2.potential + homest3.potential + homest4.potential +
                homest5.potential + homest6.potential + homest7.potential + homest8.potential + homest9.potential + homest10.potential + homest11.potential) AS 'sum_potential_home' FROM Match m 
                join stats_of_player homest1 on m.home_player_1 = homest1.player_api_id
                join stats_of_player homest2 on m.home_player_2 = homest2.player_api_id
                join stats_of_player homest3 on m.home_player_3 = homest3.player_api_id
                join stats_of_player homest4 on m.home_player_4 = homest4.player_api_id
                join stats_of_player homest5 on m.home_player_5 = homest5.player_api_id
                join stats_of_player homest6 on m.home_player_6 = homest6.player_api_id
                join stats_of_player homest7 on m.home_player_7 = homest7.player_api_id
                join stats_of_player homest8 on m.home_player_8 = homest8.player_api_id
                join stats_of_player homest9 on m.home_player_9 = homest9.player_api_id
                join stats_of_player homest10 on m.home_player_10 = homest10.player_api_id
                join stats_of_player homest11 on m.home_player_11 = homest11.player_api_id;
CREATE TABLE IF NOT EXISTS Away_Potential AS SELECT m.Id , (awayst1.potential + awayst2.potential + awayst3.potential + awayst4.potential +
                awayst5.potential + awayst6.potential + awayst7.potential + awayst8.potential + awayst9.potential + awayst10.potential + awayst11.potential) AS 'sum_potential_away' FROM Match m 
                join stats_of_player awayst1 on m.away_player_1 = awayst1.player_api_id
                join stats_of_player awayst2 on m.away_player_2 = awayst2.player_api_id
                join stats_of_player awayst3 on m.away_player_3 = awayst3.player_api_id
                join stats_of_player awayst4 on m.away_player_4 = awayst4.player_api_id
                join stats_of_player awayst5 on m.away_player_5 = awayst5.player_api_id
                join stats_of_player awayst6 on m.away_player_6 = awayst6.player_api_id
                join stats_of_player awayst7 on m.away_player_7 = awayst7.player_api_id
                join stats_of_player awayst8 on m.away_player_8 = awayst8.player_api_id
                join stats_of_player awayst9 on m.away_player_9 = awayst9.player_api_id
                join stats_of_player awayst10 on m.away_player_10 = awayst10.player_api_id
                join stats_of_player awayst11 on m.away_player_11 = awayst11.player_api_id;
CREATE INDEX IF NOT EXISTS stage_index ON Match (stage);
CREATE INDEX IF NOT EXISTS stage_index ON Match (home_team_api_id); 
CREATE INDEX IF NOT EXISTS stage_index ON Match (away_team_api_id);                 
CREATE TABLE IF NOT EXISTS Trend_home_team AS SELECT m.Id first.home_team_goal second.home_team_goal third.home_team_goal first.away_team_goal second.away_team_goal thir.away_team_goal from Match m
                join Match first_home on first_home.stage = (m.stage - 1) AND (first.home_team_api_id = m.home_team_api_id)
                join Match first_away on first_away.stage = (m.stage - 1) AND (first.away_team_api_id = m.home_team_api_id)
                join Match second_home on second_home.stage = (m.stage - 1) AND (second.home_team_api_id = m.home_team_api_id)
                join Match second_away on second_away.stage = (m.stage - 1) AND (second.away_team_api_id = m.home_team_api_id)
                join Match third_home on third_home.stage = (m.stage - 1) AND (third.home_team_api_id = m.home_team_api_id)
                join Match third_away on third_away.stage = (m.stage - 1) AND (third.away_team_api_id = m.home_team_api_id)
CREATE INDEX IF NOT EXISTS away_potential_index ON Away_Potential (id);
"""
c.executescript(select_script)

#print(df.columns.values)
selectString = """SELECT (sum_potential_home - sum_potential_away) AS potential_diff, home_team_goal, away_team_goal FROM Home_Potential h JOIN Away_Potential a ON h.id = a.Id"""
selectStringTrendHome = """SELECT * FROM Trend_home"""

trend_df = pd.read_sql_query(selectStringTrendHome, con)
              
features = pd.read_sql_query(selectString, con)

#Remove non-finite values and set new index
features = features[np.isfinite(features['potential_diff'])].reset_index()


#Convert Dataframe into pandas arrays 

potential= features.as_matrix(columns=['potential_diff']).flatten()
target = np.empty(potential.size)

##home team wins: 1, away team wins: -1, draw: 0
for index, item in enumerate(features['home_team_goal']):
    if(item>features['away_team_goal'][index]):
        target[index]=1
    elif(item<features['away_team_goal'][index]):
        target[index]=-1
    else:
        target[index]=0


print potential
print target

#calculate score without cross-validation
X_train, X_test, y_train, y_test = train_test_split(potential, target, test_size=0.2, random_state=0)

#for single feature reshape
X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)

clf = tree.DecisionTreeClassifier();
clf.fit(X_train, y_train)     

scoreSingleSplit=clf.score(X_test, y_test)

#Calculate score with cross-validation
potential= potential.reshape(-1,1)
crossValScore=cross_val_score(clf, potential, target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (crossValScore.mean(), crossValScore.std() * 2))
con.close()