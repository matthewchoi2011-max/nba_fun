import pandas as pd
import random
import math

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
        self.two_point_tendency = (self.FGM - self.FG3M) / self.FGA if self.FGA > 0 else 0
        self.three_point_tendency = self.FG3M / self.FGA if self.FGA > 0 else 0

        ''' 
        self.FG3_tendency = 0
        self.two_point_tendency = 0
        self.block_tendency = 0
        self.steal_tendency = 0
        self.rebound_tendency = 0
        self.assist_tendency = 0
        
        self.minutes = player_df['minutes']


def calculate_tendencies(tendency, nba_df):


    nba_df['years_norm'] = nba_df['season_GP']/nba_df['season_GP'].max()
    max_years = max(nba_df['seasons_played'])

    #weights for FG tendencies
    three_w1 = 0.45
    three_w2 = 0.45
    three_w3 = 0.1

    max_3pt_score = max(nba_df['season_FGA'] * nba_df['season_FG_PCT'])
    max_log_3PM = max(math.log1p(x) for x in nba_df['season_FGM'])

    #weights for 3PT FG tendencies
    fg_w1 = 0.5
    fg_w2 = 0.4
    fg_w3 = 0.1

    max_fg_score = max(nba_df['season_3PA'] * nba_df['season_3P_PCT'])
    max_log_FGM = max(math.log1p(x) for x in nba_df['season_3PM'])

    #weights for rebounding tendencies
    rb_w1 = 0.5
    rb_w2 = 0.4
    rb_w3 = 0.1

    max_total_rebounds = max(nba_df['TRB'])
    max_log_career_rb = max(math.log1p(x) for x in nba_df['season_RBG'])


    #weights for assist tendencies
    ast_w1 = 0.5
    ast_w2 = 0.4
    ast_w3 = 0.1

    max_total_assists = max(nba_df['TAST'])
    max_log_career_ast = max(math.log1p(x) for x in nba_df['season_APG'])

    #weights for steal tendencies
    stl_w1 = 0.5
    stl_w2 = 0.4
    stl_w3 = 0.1

    max_total_steals = max(nba_df['TSTL'])
    max_log_career_stl = max(math.log1p(x) for x in nba_df['season_SPG'])

    #weights for block tendencies
    blk_w1 = 0.5
    blk_w2 = 0.4
    blk_w3 = 0.1

    max_total_blks = max(nba_df['TBLK'])
    max_log_career_blk = max(math.log1p(x) for x in nba_df['season_BPG'])

    #compute weighted scores
    for index,player in nba_df.iterrows():

        three_norm = player['T3PM']/max_log_3PM
        fg_norm = player['TFGM']/max_fg_score
        rb_norm = player['TRB']/max_total_rebounds
        ast_norm = player['TAST']/max_total_assists
        stl_norm = player['TSTL']/max_total_steals
        blk_norm = player['TBLK']/max_total_blks

        three_score = three_w1 * three_norm + three_w2 * 
        
   


    
    
    


    


    