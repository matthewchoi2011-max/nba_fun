#from Player_extract import season_stats
from numpy.ma.extras import average
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from xgboost import XGBClassifier
#import xgboost as xgb
from pathlib import Path
import time
import requests
import json
import os
import numpy as np
import pandas as pd
import random
#from sklearn.feature_selection import mutual_info_classif

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

    specs = player['specs']
    specs_columns = {k: v for k, v in specs.items() if k != 'teams'}

    seasons_played = len(player['season_stats'])

    #change season_stats and playoff stats to the MAX, and
    season_stats = player.get('season_stats', {})
    playoff_stats = player.get('playoff_stats', {})

    season_stat = collect_stats(season_stats)
    playoff_stat =  collect_stats(playoff_stats)
    #print("season_stat: ", season_stat)

    sum_stats = ['FGM','FGA','FG3M','FG3A','FTM','FTA','GP']

    season_max_stats = {}
    playoff_max_stats = {}

    for stat,val in season_stat.items():
        if stat in sum_stats:
            season_max_stats[stat] = sum(val)
        else:
            season_max_stats[stat] = average(val)
    
    for stat,val in playoff_stat.items():
        if stat in sum_stats:
            playoff_max_stats[stat] = sum(val)
        else:
            playoff_max_stats[stat] = average(val)


    revised_player = {
        "player_id": player['player_id'],
        "full_name": player['full_name'],
        "seasons_played": seasons_played,
        **award_count, **specs_columns
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
nba_dataset.to_csv("nba_dataset.csv", sep=',', index=False)