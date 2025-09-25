#from Player_extract import season_stats
from numpy.ma.extras import average
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
from pathlib import Path
import time
import requests
import json
import os
import numpy as np
import pandas as pd
import random
from sklearn.feature_selection import mutual_info_classif

'''
encode the Json files, by importing said Json files for each player, and
retrieving a dataframe for each player

Store the dataframes of every player into a new dataset

Retrieve 300 players(50 of who are in the top 75 and 250 of whom are not) as a training dataset

Test on entire dataset
'''



def collect_stats(stats_dict):
    stat_values = {}

    for season, stats in stats_dict.items():
        for stat, val in stats.items():
            # Filter out NaN and None
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                stat_values.setdefault(stat, []).append(val)

    return stat_values

def convert_player_dataframe(player):
    #convert json to dataframe
    #convert all awards to be completely numerical

    awards_dict = player['awards']
    award_count = {award: len(years) for award, years in awards_dict.items()}

    seasons_played = len(player['season_stats'])

    #change season_stats and playoff stats to the MAX, and
    season_stats = player.get('season_stats', {})
    playoff_stats = player.get('playoff_stats', {})

    season_stat = collect_stats(season_stats)
    playoff_stat =  collect_stats(playoff_stats)

    season_max_stats = {stat: average(vals) for stat, vals in season_stat.items()}
    playoff_max_stats = {stat: average(vals) for stat, vals in season_stat.items()}


    revised_player = {
        "player_id": player['player_id'],
        "full_name": player['full_name'],
        "seasons_played": seasons_played,
        **award_count
    }
    for k, v in season_max_stats.items():
        revised_player[f"season_{k}"] = v

    # Prefix playoff stats with "playoff_"
    for k, v in playoff_max_stats.items():
        revised_player[f"playoff_{k}"] = v


    player_df = pd.DataFrame([revised_player])
    return player_df




#retrieve all json files in players folder!
folder_path = "players/"
nba_dataset = pd.DataFrame()

#add ALL player data into nba_dataset
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename),encoding="utf-8") as f:
            data = json.load(f)
            #convert json to panda and add to dataset
            panda_player = convert_player_dataframe(data)
            #add to dataset
            nba_dataset = pd.concat([nba_dataset, panda_player])

nba_dataset.fillna(0, inplace=True)
nba_dataset.to_csv("nba_dataset.csv", sep="\t", index=False)


'''
#remove all players with 3 seasons or less
nba_dataset['num_seasons'] = nba_dataset['season_stats'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
nba_filtered = nba_dataset[nba_dataset['num_seasons'] > 3].drop(columns=['num_seasons'])


#remove all players with 1 playoff series or less
nba_filtered['num_seasons'] = nba_filtered['playoff_stats'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
nba_filtered = nba_filtered[nba_filtered['num_seasons'] > 1].drop(columns=['num_seasons'])

print(nba_filtered)

#remove all players with 100 games in the regular season or less
def total_games_played(season_stats):
    return sum(
        season.get("GP", 0)
        for season in season_stats.values()
        if isinstance(season, dict) and isinstance(season.get("GP", 0), (int, float))
        )

# Apply the function to create a new column
nba_filtered["total_GP"] = nba_filtered["season_stats"].apply(total_games_played)
# Filter rows where total_GP >= 100
valid_players = nba_filtered[nba_filtered["total_GP"] >= 100].reset_index(drop=True)
'''

#RETRIEVE TOP 75 PLAYERS AND ADD TOP 75 VARIABLE TO DATAFRAME
nba_75_players = [
    "Kareem Abdul-Jabbar", "Ray Allen", "Giannis Antetokounmpo", "Carmelo Anthony",
    "Nate Archibald", "Paul Arizin", "Charles Barkley", "Rick Barry", "Elgin Baylor",
    "Dave Bing", "Larry Bird", "Kobe Bryant", "Wilt Chamberlain", "Bob Cousy",
    "Dave Cowens", "Billy Cunningham", "Stephen Curry", "Anthony Davis", "Dave DeBusschere",
    "Clyde Drexler", "Tim Duncan", "Kevin Durant", "Julius Erving", "Patrick Ewing",
    "Walt Frazier", "Kevin Garnett", "George Gervin", "Hal Greer", "James Harden",
    "John Havlicek", "Elvin Hayes", "Allen Iverson", "LeBron James", "Magic Johnson",
    "Sam Jones", "Michael Jordan", "Jason Kidd", "Kawhi Leonard", "Damian Lillard",
    "Jerry Lucas", "Karl Malone", "Moses Malone", "Pete Maravich", "Bob McAdoo",
    "Kevin McHale", "George Mikan", "Reggie Miller", "Earl Monroe", "Steve Nash",
    "Dirk Nowitzki", "Shaquille O'Neal", "Hakeem Olajuwon", "Robert Parish", "Chris Paul",
    "Gary Payton", "Bob Pettit", "Paul Pierce", "Scottie Pippen", "Willis Reed",
    "Oscar Robertson", "David Robinson", "Dennis Rodman", "Bill Russell", "Dolph Schayes",
    "Bill Sharman", "John Stockton", "Isiah Thomas", "Nate Thurmond", "Wes Unseld",
    "Dwyane Wade", "Bill Walton", "Jerry West", "Russell Westbrook", "Lenny Wilkens",
    "Dominique Wilkins", "James Worthy"
]

#feature = print(list(nba_dataset.columns))

valid_players = nba_dataset

#adds is_top_75 to player dataframe which returns 1 if in nba_75_players and 0 if otherwise
valid_players['is_top_75'] = valid_players['full_name'].apply(lambda name: 1 if name in nba_75_players else 0)
valid_players.to_csv("nba_filtered_dataset.txt", sep="\t", index=False)

#features
feature_columns = ["NBA All-Star", "MVP", "All-NBA-First", "All-NBA-Second", "All-NBA-Third",
    "DPOY", "All-Defensive-First", "All-Defensive-Second", "Championships",
    "FMVP", "Olympic Gold", "Olympic Silver", "Olympic Bronze",'season_PPG', 'season_APG',
    'season_RBG', 'season_SPG', 'season_BPG', 'season_TPG', 'season_MPG', 'season_FG_PCT',
    'season_FG3_PCT', 'season_FT_PCT', 'season_PM', 'playoff_PPG', 'playoff_APG',
    'playoff_RBG', 'playoff_SPG', 'playoff_BPG', 'playoff_TPG', 'playoff_MPG', 'playoff_FG_PCT',
    'playoff_FG3_PCT', 'playoff_FT_PCT',  'playoff_PM']






def main():
    # ML MODEL
    X = valid_players[feature_columns]

    y = valid_players['is_top_75']
    #print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf = xgb.XGBClassifier(
        objective="binary:logistic",  # since we are doing classification
        eval_metric="logloss",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )

    # model.fit(X_train, y_train)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    #print("Accuracy:", accuracy_score(y_test, y_pred))
    mi_scores = mutual_info_classif(X, y)
    #print(mi_scores, mi_scores)

    correlations = valid_players.corr(numeric_only=True)["is_top_75"].drop("is_top_75").sort_values(key=abs,
                                                                                                    ascending=False)
    #print("Most correlated features with is_top_75:")
    #print(correlations.head(10))  # Top 10 most correlated

    # DWIGHT HOWARD TEST
    dwight = valid_players[valid_players['full_name'] == 'Damian Lillard']
    test = dwight[feature_columns]
    prediction = clf.predict(test)
    #print(prediction)

    # model top 75
    probs = clf.predict_proba(X)[:, 1]
    valid_players['predicted_prob'] = probs
    top_75_predicted = valid_players.sort_values(by='predicted_prob', ascending=False).head(75)
    list = []
    for idx, row in top_75_predicted.iterrows():
        #print(f"{row['full_name']}: {row['predicted_prob']:.4f}")
        list.append(row['full_name'])

    return list


if __name__ == "__main__":
    result = main()
    print(result)




