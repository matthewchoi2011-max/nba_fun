import torch
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import xgboost
import pandas
import numpy as np
import time
import json
import random

#import player jsons from player folder
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

nba_dataset.to_csv("nba_dataset.txt", sep="\t", index=False)








