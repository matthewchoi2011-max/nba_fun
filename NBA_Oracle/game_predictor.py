#import json
import pandas as pd
import random
import math

#get all players from nba_dataset.csv 
players_df = pd.read_csv('nba_dataset.csv', encoding='latin1',sep='\t')

#convert csv to be comma space instead

#retrieve all column names
#print(players_df.columns)
#print("\n")

#make a team with 15 different players and assign minutes to them accordingly

def make_team():
    total_minutes = 240 #5 players * 48 minutes
    team = []
    team_stats = {}

    while len(team) < 15:
        player = random.choice(players_df['player_id'].tolist())
        if player not in team:
            team.append(player)
            player_stats = players_df[players_df['player_id'] == player].to_dict(orient='records')[0]
            team_stats[player] = player_stats
            
    #print(team_stats)
    new_team_stats = pd.DataFrame.from_dict(team_stats, orient='index')
    #new_team_stats.to_csv('team_stats.csv', index=False)
    return new_team_stats

def determine_minutes(team_stats):
 
    #print(team_stats)
    
    # assign more minutes to players with higher award metrics, added weights for each award
    for index, player in team_stats.iterrows():

        player_merit_sum = (math.sqrt(player['seasons_played']) +
                player['NBA All-Star'] * 2 +   
                player['All-NBA-First'] * 4.5 + 
                player['All-NBA-Second'] * 3.5 + 
                player['All-NBA-Third'] * 2.5 +
                player['Championships'] * 2 +
                player['MVP'] * 5 +
                player['FMVP'] * 4 +  
                player['DPOY'] * 2.5 +
                player['All-Defensive-First'] * 1.8 +
                player['All-Defensive-Second'] * 1.3)
        
        #add player_merit to team_stats
        team_stats.at[index,'player_merit'] = player_merit_sum

        #assume 2.33 points per assist
        points_generated = (player['season_PPG'] * player['season_GP'] + 
                            player['season_APG'] * player['season_GP'] * 2.33)
        
        #assume each rebound generates 1.15 points
        rebeounds_generated = (player['season_RBG'] * player['season_GP'] * 1.15) 
        total_contribution = points_generated + rebeounds_generated

        #assume each steal generates 2.25 points, each block generates 2.25 points
        defensive_contribution = (player['season_SPG'] * player['season_GP'] * 2.25) + (player['season_BPG'] * player['season_GP'] * 2)

        losses_due_to_turnovers = (player['season_TPG'] * player['season_GP'] * 2.25)

        #net contribution = total contribution + defensive contribution - losses due to turnovers
        net_contribution = total_contribution + defensive_contribution - losses_due_to_turnovers
        team_stats.at[index,'player_merit'] = net_contribution * player_merit_sum


    #sort team_stats by player_merit
    team_stats.sort_values(by='player_merit', ascending=False, inplace=True)
    #print(team_stats[['full_name','player_merit']])

    #using player merit, assign minutes accordingly

    '''
    Expected conversion:

    LeGOAT = 650k

    MJ = 563k

    Kobe Bryant: 391934.594 = superstar (40+ minutes)

    300k+ = top 20 players of all time

    Carmelo Anthony: 90909 = star (30-40 minutes)

    Tony Parker: 81505.441 = star (30-40 minutes)

    Pau Gasol: 74265.2 = star (30-40 minutes)

    Devin Booker = 45635.7 = all star (30-40 minutes)

    Sam Cassell: 30267 = all star (30-40 minutes)

    15-40k = bone fide all star, not the most consistent

    Al Jefferson: 11789 = solid starter (20-30 minutes)

    Jamal Murray: 10509.370 = solid starter (20-30 minutes)

    Jose Calderon 5419.339 = role player (10-20 minutes)

    Eric Gordon 5137.339 = role player (10-20 minutes)

    Bismack Biyombo: 3315.5 = role player/bench (10-20 minutes)

    Delonte West: 1000 ish, bench

    < 10000 = bench (0-10 minutes)

    '''

    total_merit = team_stats['player_merit'].sum()

    #calculate minutes based on player merit/total merit of team
    for index, player in team_stats.iterrows():
        ratioed_merit = player['player_merit'] / total_merit

        low = 1
        high = 10
        population = list(range(low, high + 1))
        weights = [x for x in population]  # Medium weight for all players

        if(ratioed_merit > 0.2):
            
            low = 32
            high = 45
            population = list(range(low, high + 1))
            weights = [x for x in population]  # Higher numbers get more weight
            player_minutes = random.choices(population, weights=weights, k=1)[0]

        if(ratioed_merit < 0.2 and ratioed_merit > 0.1):
            low = 20
            high = 32
            population = list(range(low, high + 1))
            weights = [x for x in population]  # Higher numbers get more weight
            player_minutes = random.choices(population, weights=weights, k=1)[0]

        elif(ratioed_merit < 0.1 and ratioed_merit > 0.03):
            low = 10
            high = 20
            population = list(range(low, high + 1))
            weights = [x for x in population]  # Higher numbers get more weight
            player_minutes = random.choices(population, weights=weights, k=1)[0]

        elif(ratioed_merit < 0.03):
            low = 0
            high = 10
            population = list(range(low, high + 1))
            weights = [x for x in population]  # Higher numbers get more weight
            player_minutes = random.choices(population, weights=weights, k=1)[0]

        team_stats.at[index,'minutes'] = round(player_minutes,1)


    #scale minutes to total 240
    total_minutes = team_stats['minutes'].sum()
    scale = total_minutes / 240 
    team_stats['minutes'] = round(team_stats['minutes']/scale,1)



    #readjust to keep all players below 45 minutes

    # Robustly cap at 45 and redistribute deficit to those under 40
    team_stats['minutes'] = team_stats['minutes'].clip(upper=45)
    # Redistribute until total is close to 240 or no one is under 40
    for _ in range(20):  # limit iterations to avoid infinite loop
        total = team_stats['minutes'].sum()
        deficit = 240 - total
        under = team_stats['minutes'] < 40
        if abs(deficit) < 0.01 or not under.any():
            break
        add_per_player = deficit / under.sum()
        team_stats.loc[under, 'minutes'] += add_per_player
        team_stats['minutes'] = team_stats['minutes'].clip(upper=45)
    team_stats['minutes'] = team_stats['minutes'].round(1)


    return team_stats


class Player:
    def __init__(self,player_df):
        self.name = player_df['full_name']
        self.id = player_df['player_id']
        self.ppg = player_df['season_PPG']
        self.apg = player_df['season_APG']
        self.rpg = player_df['season_RBG']
        self.spg = player_df['season_SPG']
        self.bpg = player_df['season_BPG']
        self.tpg = player_df['season_TPG']
        self.gp = player_df['season_GP']
        self.FGM = player_df['season_FGM']
        self.FGA = player_df['season_FGA']
        self.FG_PCT = player_df['season_FG_PCT']
        self.FG3M = player_df['season_3PM']
        self.FG3A = player_df['season_3PA']
        self.FG3_PCT = player_df['season_3P_PCT']
        self.FTM = player_df['season_FTM']
        self.FTA = player_df['season_FTA']
        self.FT_PCT = player_df['season_FT_PCT']
        
        #calculate tendencies
        '''
            scoring tendency + passing tendency must be add up to 100%

            stealing tendency + blocking tendency must add up to 100%

            turnover tendency will be based on 

            fouling tendency will be based on minutes played and defensive stats!

        
        ''' 
        



        self.rebound_tendency = 0
        self.steal_tendency = 0
        self.block_tendency = 0
        self.turnover_tendency = 0
        self.possessive_tendency = 0


        self.minutes = player_df['minutes']


class Team:
    def __init__(self,team_df):
        self.players = []
        for index, player in team_df.iterrows():
            self.players.append(Player(player))


        self.starters = self.players[:5]
        self.bench = self.players[5:]

        

class Game:
    def __init__(self,team1,team2,minutes):
        self.team1 = Team(team1)
        self.team2 = Team(team2)
        self.minutes = minutes

    def tip_off(self):
        #select the tallest players from each starting lineup
        


if __name__ == "__main__":
    team1 = make_team()
    team2 = make_team()
    #print(team1)
    #print(team2)
    sorted_team1 = determine_minutes(team1)
    sorted_team2 = determine_minutes(team2)

    for player in sorted_team1['full_name']:
        Player = Player(player['player_id'], player['full_name'], 
                        player['season_PPG'], player['season_APG'], player['season_RBG'], 
                        player['season_SPG'], player['season_BPG'], player['season_TPG'], 
                        player['season_GP'])


    #simulate_game(sorted_team1, sorted_team2,48)



        

   




