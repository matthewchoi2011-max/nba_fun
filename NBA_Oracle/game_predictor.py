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


class Player_Box:
    def __init__(self, player_id):
        self.player_id = player_id
        self.points = 0
        self.rebounds = 0
        self.assists = 0
        self.steals = 0
        self.blocks = 0
        self.turnovers = 0
        self.fouls = 0
        self.minutes_played = 0
        self.FG_PCT = 0.0  # Field Goal Percentage
        self.threeP_PCT = 0.0  # Three-Point Percentage
        self.FT_PCT = 0.0  # Free Throw Percentage
        self.FG_attempts = 0
        self.threeP_attempts = 0
        self.FT_attempts = 0

    def update_stats(self, points=0, rebounds=0, assists=0, steals=0, blocks=0, turnovers=0, fouls=0, minutes=0, FG_attempts=0, FG_made=0, threeP_attempts=0, threeP_made=0, FT_attempts=0, FT_made=0):
        self.points += points
        self.rebounds += rebounds
        self.assists += assists
        self.steals += steals
        self.blocks += blocks
        self.turnovers += turnovers
        self.fouls += fouls
        self.minutes_played += minutes
        self.FG_attempts += FG_attempts
        self.threeP_attempts += threeP_attempts
        self.FT_attempts += FT_attempts

        # Update shooting percentages
        if self.FG_attempts > 0:
            self.FG_PCT = (self.FG_PCT * (self.FG_attempts - FG_attempts) + FG_made) / self.FG_attempts
        if self.threeP_attempts > 0:
            self.threeP_PCT = (self.threeP_PCT * (self.threeP_attempts - threeP_attempts) + threeP_made) / self.threeP_attempts
        if self.FT_attempts > 0:
            self.FT_PCT = (self.FT_PCT * (self.FT_attempts - FT_attempts) + FT_made) / self.FT_attempts

class Game:
    def __init__(self, team1, team2, minutes):
        self.team1 = team1
        self.team2 = team2
        self.game_minutes = minutes
        self.quarter_length = minutes / 4  # assuming 4 quarters
        self.team1_score = 0
        self.team2_score = 0
        self.team1_stats = {}
        self.team2_stats = {}
        self.team1_currently_on_court = []
        self.team2_currently_on_court = []
        self.possession = None  # 'team1' or 'team2'
        self.timer = 0  # in seconds
        self.overtime = False
        self.overtime_counter = 0
        self.team1_box = None
        self.team2_box = None

    def start_box(self,team1, team2):

        list1 = team1['player_id'].tolist()
        list2 = team2['player_id'].tolist()

        team1_box = {}
        team2_box = {}  
        for player in list1:
            team1_box[player] = Player_Box(player)
        
        for player in list2:
            team2_box[player] = Player_Box(player)

        self.team1_box = team1_box
        self.team2_box = team2_box


    def start_game(self):
        # two starting centers for each team begin tip off
        self.possession = random.randomly.choice(['team1', 'team2'])
        return self.possession
    
    def end_game(self):
        # team 1 wins
        if(self.timer >= self.minutes * 60 and self.team1_score > self.team2_score):
            return "Team 1 wins!"
        
        # team 2 wins
        elif(self.timer >= self.minutes * 60 and self.team2_score > self.team1_score):
            return "Team 2 wins!"
        
        # overtime
        elif(self.timer >= self.minutes * 60 and self.team1_score == self.team2_score):
            #initialize new game for overtime
            return  "Overtime!"
        

    def play(self,team1,team2):
        #time left on play clock for possession
        possession_max = 24 # seconds
        if(720 - self.timer < possession_max):
            possession_time = 720 - self.timer

        #begin play for possession



            
    def possession(self):
        # Simulate a single possession for the team with the ball

        if(self.possession == 'team1'):
            #team 1 has the ball
            # play(team1,team2)
            pass
        elif(self.possession == 'team2'):
            #team 2 has the ball
            # play(team2,team1)
            pass


    def substitute_player(self, team, player_out_id, player_in_id):
        # Substitute a player in and out of the game for the specified team
        if(team == 'team1'):
            if player_out_id in self.team1_currently_on_court:
                self.team1_currently_on_court.remove(player_out_id)
                self.team1_currently_on_court.append(player_in_id)
        elif(team == 'team2'):
            if player_out_id in self.team2_currently_on_court:
                self.team2_currently_on_court.remove(player_out_id)
                self.team2_currently_on_court.append(player_in_id)
        

def simulate_game(team1_df, team2_df, minutes):
    #simulate a game between team1 and team2 based on player stats and minutes played
    
    #GAME START
    team1_score = 0
    team2_score = 0
    
    #Choose 5 players from each team to start
    team1_starters = team1_df.head(5)
    team2_starters = team2_df.head(5)

    print("Team 1 Starters:")
    print(team1_starters['full_name'])

    print("\nTeam 2 Starters:")
    print(team2_starters['full_name'])


if __name__ == "__main__":
    team1 = make_team()
    team2 = make_team()
    #print(team1)
    #print(team2)
    sorted_team1 = determine_minutes(team1)
    sorted_team2 = determine_minutes(team2)
    simulate_game(sorted_team1, sorted_team2,48)



        

   




