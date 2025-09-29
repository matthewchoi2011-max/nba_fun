

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

        self.current_game_stats = None


    def sub_out(self,Player1, Player2):
        #remove P1 from starting lineup and move to bench
        self.starters.pop(Player(Player1))
        self.bench.append(Player(Player1))
        #remove P2 from bench lineup and move to starting lineup
        self.starters.pop(Player(Player2))
        self.bench.append(Player(Player2))        



    