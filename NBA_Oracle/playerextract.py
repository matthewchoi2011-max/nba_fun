from sys import set_asyncgen_hooks
import sys
from nba_api.stats.static import players
from nba_api.stats.endpoints import playerawards, commonplayerinfo, playercareerstats
from nba_api.stats.endpoints import playergamelog
import time
import requests
import json
import os
import io
import numpy as np
from json.decoder import JSONDecodeError


'''
Retrieve the stats of every NBA player and their accolades and then stores them in a JSON file

In the following order:
{
    Per Season AND Playoffs:
    [
        Points per game(PPG),
        Assists per game(APG),
        Rebounds per game(RBG),
        Steals per game(SPG),
        Blocks per game(BPG),
        Turnovers per game(TPG),
        Plus/Minus(PM),
        Minutes per game(MPG),
        Field Goal Percentage(FG_PCT),
        3pt Field Goal Percentage(FG3_PCT),
        Free Throw Percentage(FT_PCT),
        Total Games played(G),
    ],
    Awards
    [
        NBA All-Star appearances:{},
        NBA All-Star MVPs:{},
        MVPs:{},
        All nba appearances(1st, 2nd, 3rd):{},
        DPOYs:{},
        All defence appearances(1st, 2nd):{},
        Championships:{},
        FMVPs:{}
        Olympic Medals:{}
        Playoff Appearance/Total Appearances score:{}
    ]
}
'''

def plus_minus(player_id,season_id,season_type):
    time.sleep(1)
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season_id, season_type_all_star=season_type)
    # Extract DataFrame
    df = gamelog.get_data_frames()[0]
    # Get Total and Average PM
    total_pm = df['PLUS_MINUS'].sum()
    average_pm = df['PLUS_MINUS'].mean()
    return average_pm

def career_stats(season_stats,season_type,player_data):

    key = ""
    if season_type == 'Playoffs':
        key = "playoff_stats"
    elif season_type == 'Regular Season':
        key = "season_stats"
    for _, season in season_stats.iterrows():
        season_id = season["SEASON_ID"]
        gp = season.get('GP') or 0
        player_data[key][season_id] = {}

        def safe_div(numerator, denominator):
            try:
                return round((numerator or 0) / (denominator or 1), 1) if denominator else 0
            except:
                return 0

        def safe_round(value, digits):
            try:
                return round(value, digits)
            except:
                return 0

        player_data[key][season_id]["PPG"] = safe_div(season.get('PTS'), gp)
        player_data[key][season_id]["APG"] = safe_div(season.get('AST'), gp)
        player_data[key][season_id]["RBG"] = safe_div(season.get('REB'), gp)
        player_data[key][season_id]["SPG"] = safe_div(season.get('STL'), gp)
        player_data[key][season_id]["BPG"] = safe_div(season.get('BLK'), gp)
        player_data[key][season_id]["TPG"] = safe_div(season.get('TOV'), gp)

        # Plus-minus might be a custom function that fails
        try:
            pm = plus_minus(player_data["player_id"], season_id, season_type)
            player_data[key][season_id]["PM"] = safe_round(pm, 1)
        except:
            player_data[key][season_id]["PM"] = 0

        player_data[key][season_id]["MPG"] = safe_div(season.get('MIN'), gp)
        player_data[key][season_id]["FG_PCT"] = safe_round(season.get('FG_PCT') or 0, 3)
        player_data[key][season_id]["FG3_PCT"] = safe_round(season.get('FG3_PCT') or 0, 3)
        player_data[key][season_id]["FT_PCT"] = safe_round(season.get('FT_PCT') or 0, 3)
        player_data[key][season_id]["GP"] = gp

    return player_data

def accolades(awards_df, player_data):
    # Initialize award categories
    award_keys = {
        "NBA All-Star": [],
        "NBA All-Star Game MVP": [],
        "MVP": [],
        "All-NBA-First": [],
        "All-NBA-Second": [],
        "All-NBA-Third": [],
        "DPOY": [],
        "All-Defensive-First": [],
        "All-Defensive-Second": [],
        "Championships": [],
        "FMVP": [],
        "Olympic Gold": [],
        "Olympic Silver": [],
        "Olympic Bronze": [],
    }

    player_data["awards"] = award_keys.copy()

    for _, award in awards_df.iterrows():
        desc = award.get("DESCRIPTION") or award.get("Description")
        season = award.get("SEASON")
        team_number = str(award.get("ALL_NBA_TEAM_NUMBER"))

        if desc == "NBA All-Star":
            player_data["awards"]["NBA All-Star"].append(season)
        elif desc == "NBA All-Star Most Valuable Player":
            player_data["awards"]["NBA All-Star Game MVP"].append(season)
        elif desc == "NBA Most Valuable Player":
            player_data["awards"]["MVP"].append(season)
        elif desc == "All-NBA":
            if team_number == "1":
                player_data["awards"]["All-NBA-First"].append(season)
            elif team_number == "2":
                player_data["awards"]["All-NBA-Second"].append(season)
            elif team_number == "3":
                player_data["awards"]["All-NBA-Third"].append(season)
        elif desc == "NBA Defensive Player of the Year":
            player_data["awards"]["DPOY"].append(season)
        elif desc == "All-Defensive Team":
            if team_number == "1":
                player_data["awards"]["All-Defensive-First"].append(season)
            elif team_number == "2":
                player_data["awards"]["All-Defensive-Second"].append(season)
        elif desc == "NBA Champion":
            player_data["awards"]["Championships"].append(season)
        elif desc == "NBA Finals Most Valuable Player":
            player_data["awards"]["FMVP"].append(season)
        elif desc == "Olympic Gold Medal":
            player_data["awards"]["Olympic Gold"].append(season)
        elif desc == "Olympic Silver Medal":
            player_data["awards"]["Olympic Silver"].append(season)
        elif desc == "Olympic Bronze Medal":
            player_data["awards"]["Olympic Bronze"].append(season)

    return player_data


def attributes(attributes_df, player_data):

    attribute = {
        "Height": '',
        "Weight": '',
        "Position" : '',
        "Teams": [],
        "DraftRound":'',
        "DraftNumber":''
    }

    player_data["specs"] = attribute.copy()
    player_data["specs"]["Height"] = attributes_df['HEIGHT']
    player_data["specs"]["Weight"] = attributes_df['WEIGHT']
    player_data["specs"]["Position"] = attributes_df['POSITION']
    player_data["specs"]["DraftRound"] = attributes_df['DRAFT_ROUND']
    player_data["specs"]["DraftNumber"] = attributes_df['DRAFT_NUMBER']





def JSON(player_data):
    folder_path = "players"  # e.g., relative path inside your project
    os.makedirs(folder_path, exist_ok=True)  # create folder if it doesn't exist

    # Define full file path
    file_path = os.path.join(folder_path, f"{player_data['full_name'].replace(' ', '_')}.json")

    # save file as JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(player_data, f, indent=4, ensure_ascii=False)


def chunks(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# Your full player list


if len(sys.argv) < 2:
    print("No argument provided.")
else:
    i = int(sys.argv[1])
    print(f"Running with number: {i}")

player_list = players.get_players()
chunk_size = 30     # number of players per batch
batches = chunks(player_list, chunk_size)


#i = 1  # Change this to the segment number you want (e.g., 0-based index)

if i >= len(batches):
    print(f"Segment {i} is out of range (only {len(batches)} total).")
else:
    segment = batches[i]
    print(f"Processing segment {i + 1} out of {len(batches)}")


for player in segment:
    
    player_id = player['id']
    player_name = player['full_name']
    time.sleep(1.5)  # avoid rate-limiting
    try:
        #retrieve career dataframe
        career = playercareerstats.PlayerCareerStats(player_id, timeout=600)
        #retrieve award dataframe
        time.sleep(1.5)
        awards = playerawards.PlayerAwards(player_id=player_id)


        player_data = {
            "player_id": player_id,
            "full_name": player_name,
            "awards": {},
            "season_stats": {},
            "playoff_stats": {},
            "specs":{}
        }
        #fill in attributes
        time.sleep(1.5)
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        playerinfo = info.get_normalized_dict()['CommonPlayerInfo'][0]
        #print(playerinfo)
        attributes(playerinfo, player_data)
        #teams played for
        player_data["specs"]["Teams"] = career.get_data_frames()[0]['TEAM_ABBREVIATION'].unique().tolist()


        season_stats = career_stats(career.get_data_frames()[0], 'Regular Season', player_data)
        time.sleep(1.5)
        playoff_stats = career_stats(career.get_data_frames()[2], 'Playoffs', player_data)
        time.sleep(1.5)
        accolades(awards.get_data_frames()[0], player_data)
        time.sleep(1.5)
        
        JSON(player_data)
        print(f"extracted {player_data['full_name']}\n")
    except Exception as e:
        print(f"Error processing {player_name}: {e}")


