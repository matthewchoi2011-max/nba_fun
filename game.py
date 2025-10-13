import random


class PlayerBoxScore:
    def __init__(self, playername, playerid):
        self.playerid = playername
        self.name = playerid
        self.points = 0
        self.assists = 0
        self.rebounds = 0
        self.steals = 0
        self.blocks = 0
        #field goals
        self.fgm = 0
        self.fga = 0
        self.fgp = 0.000
        #threes
        self.tpm = 0
        self.tpa = 0
        self.tpp = 0.000
        #free throws
        self.ftm = 0
        self.fta = 0
        self.ftp = 0.000
        #turnovers
        self.turnover = 0
        #fouls
        self.fouls = 0
        #status
        self.status = None
        self.PlusMinus = 0

    def update_percentages(self):
        if self.fga != 0:
            self.fgp = self.fgm/self.fga
        if self.tpa != 0:
            self.tpp = self.tpm/self.tpa
        if self.fta != 0:
            self.ftp = self.ftm/self.fta

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
        self.fatigue = 0
        #calculate tendencies
        '''
        self.two_point_tendency = (self.FGM - self.FG3M) / self.FGA if self.FGA > 0 else 0
        self.three_point_tendency = self.FG3M / self.FGA if self.FGA > 0 else 0

        ''' 
        self.minutes = player_df['minutes']
        self.playerboxscore = PlayerBoxScore(self.name,self.id)



class Team:
    def __init__(self,team_df):
        self.players = []
        for index, player in team_df.iterrows():
            self.players.append(Player(player))
            #

        self.starters = self.players[:5]
        self.bench = self.players[5:]
        self.current_game_stats = {}
        self.timeouts = 6
        self.streak = 0


    def sub_out(self,Player1, Player2):
        player_out = next(p for p in self.starters if p.name == Player1)
        player_in = next(p for p in self.bench if p.name == Player2)

        #add starter to bench lineup
        self.starters.remove(player_out)
        self.bench.append(player_out)

        #add bench player to starting lineup
        self.bench.remove(player_in)
        self.starters.append(player_in)


class Game:
    def __init__(self,Team1_df,Team2_df):
        self.team1 = Team(Team1_df)
        self.team2 = Team(Team2_df)
        self.possession = None
        self.time = 48 * 60 #total seconds of game is 48 minutes * 60 secs/min
    
    def jumpball(self):
        j = random.randint(1,2)
        self.possession = 1 if j == 1 else 2

    def play(self, time_left):
        
        timer = 24 #each play lasts 24 seconds, otherwise timer_left
        #define offensive team and defensive team

        #strategic timeout 



        if timer > time_left: timer = time_left

        while(timer > 0):

            timer -= random.randint(1,4)
            
            #check for steal

            #check if fouled

            #attempt shot
                #if shot is made
                #else check for rebound(either offensive or defensive)

        #shot clock violation!
        #possession = opposing team!      



    '''def simulate_possession_realistic(quarter_time_remaining, current_time):
    global possession_team, team_fouls, team_score
    shot_clock = 24
    offensive_team = team1 if possession_team=="team1" else team2
    defensive_team = team2 if possession_team=="team1" else team1

    # Strategic timeout: only if trailing by 6+ points in last 2 minutes
    if quarter_time_remaining < 120:
        trailing_team = "team1" if team_score["team1"] < team_score["team2"] else "team2"
        margin = abs(team_score["team1"] - team_score["team2"])
        if margin >= 6:
            call_timeout(trailing_team, quarter_time_remaining, current_time)

    while shot_clock > 0 and quarter_time_remaining > 0:
        # Time spent before next action
        action_time = random.randint(1,4)
        shot_clock -= action_time
        quarter_time_remaining -= action_time
        current_time += action_time

        # Select shooter
        shooter, shot_type = select_shooter(offensive_team)

        # Attempt pass
        if random.random() < shooter.get("pass_rate", 0.3):
            teammates = [p for p in offensive_team if p != shooter]
            if teammates:
                receiver = random.choice(teammates)
                play_log.append(f"{possession_team} - {shooter['name']} passes to {receiver['name']}")
                shooter = receiver

                # Time cost of pass
                pass_time = random.randint(1,3)
                shot_clock -= pass_time
                quarter_time_remaining -= pass_time
                current_time += pass_time

        # Check for steal
        if check_steal():
            play_log.append(f"{possession_team} - {shooter['name']} loses ball: STEAL!")
            possession_team = "team2" if possession_team=="team1" else "team1"
            return

        # Check for foul
        if check_foul(shooter):
            team_fouls[possession_team] += 1
            play_log.append(f"{possession_team} - {shooter['name']} fouled!")
            free_throws = 3 if shot_type=="3P" else 2
            made_ft = sum([1 for _ in range(free_throws) if random.random() < 0.75])
            team_score[possession_team] += made_ft
            play_log.append(f"{possession_team} - {shooter['name']} makes {made_ft}/{free_throws} FT")

        # Check for block
        block_prob = sum([d.get("block_rate",0.05) if shot_type=="2P" else d.get("block_rate",0.02) for d in defensive_team])
        if random.random() < block_prob:
            play_log.append(f"{possession_team} - {shooter['name']}'s shot was BLOCKED!")
            possession_team = "team2" if possession_team=="team1" else "team1"
            return

        # Attempt shot
        made = attempt_shot(shooter, shot_type)
        if made:
            points = 3 if shot_type=="3P" else 2
            team_score[possession_team] += points
            play_log.append(f"{possession_team} - {shooter['name']} scores {points} points")
            possession_team = "team2" if possession_team=="team1" else "team1"
            return
        else:
            # Rebound
            rebound = check_rebound(shooter, defensive_team)
            if rebound=="offensive":
                play_log.append(f"{possession_team} - Offensive rebound by {shooter['name']}")
                shot_clock = max(shot_clock,10)
                for p in offensive_team:
                    p["fatigue"] += 1
            elif rebound=="defensive":
                play_log.append(f"{possession_team} - Defensive rebound by opponent")
                possession_team = "team2" if possession_team=="team1" else "team1"
                return
            else:
                play_log.append(f"{possession_team} - {shooter['name']} misses and no rebound")
                for p in offensive_team:
                    p["fatigue"] += 1

    # Shot clock violation
    play_log.append(f"{possession_team} - Shot clock violation!")
    possession_team = "team2" if possession_team=="team1" else "team1"
'''



