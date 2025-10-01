#import json
import pandas as pd
import random
import math

#get all players from nba_dataset.csv 
players_df = pd.read_csv('nba_dataset.csv', encoding='utf8')

#convert csv to be comma space instead

#retrieve all column names
#print(players_df.columns)
#print("\n")

#make a team with 15 different players and assign minutes to them accordingly

def make_team(player_names=None, excluded_players=None):
    """
    Build a team from a given list of players.
    Ensures no players overlap with `excluded_players`.
    """
    if excluded_players is None:
        excluded_players = []

    total_minutes = 240  # 5 players * 48 minutes
    team = []
    team_stats = {}

    if player_names:
        for name in player_names:
            player_row = players_df.loc[players_df['full_name'] == name]
            if not player_row.empty:
                player_stats = player_row.iloc[0].to_dict()
                pid = player_stats['player_id']
                if pid not in excluded_players and pid not in team:
                    team_stats[pid] = player_stats
                    team.append(pid)
    # Fill team to 15 players
    while len(team) < 15:
        candidates = players_df[~players_df['player_id'].isin(team + excluded_players)]
        if candidates.empty:
            break
        filler = candidates['player_id'].sample(n=1).iloc[0]
        player_stats = players_df.loc[players_df['player_id'] == filler].iloc[0].to_dict()
        team_stats[filler] = player_stats
        team.append(filler)

    return pd.DataFrame.from_dict(team_stats, orient='index')



def choose_best_formation(team_stats):
    """
    Selects the best starting 5 formation based on combined player merit.
    Valid formations:
        1. G G G F F
        2. G G F F C
        3. G F F F C
    """
    # Normalize Position column
    team_stats['Position'] = team_stats['Position'].str.upper().str.strip()

    # Helper: try to select players for a given formation
    def try_formation(requirements):
        chosen = []
        remaining = team_stats.copy()
        for pos, count in requirements.items():
            candidates = remaining[remaining['Position'].str.contains(pos, na=False)]
            top = candidates.nlargest(count, 'player_merit')
            if len(top) < count:
                return None  # formation not possible
            chosen.append(top)
            remaining = remaining.drop(top.index)
        return pd.concat(chosen)

    # Define formations
    formations = [
        {"G": 3, "F": 2},          # G G G F F
        {"G": 2, "F": 2, "C": 1},  # G G F F C
        {"G": 1, "F": 3, "C": 1}   # G F F F C
    ]

    # Evaluate formations
    best = None
    best_merit = -1
    for formation in formations:
        lineup = try_formation(formation)
        if lineup is not None:
            merit_sum = lineup['player_merit'].sum()
            if merit_sum > best_merit:
                best_merit = merit_sum
                best = lineup

    return best

def determine_seconds(team_stats):
    import math
    import pandas as pd

    # --- 1. Calculate player merit ---
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

        points_generated = (player['season_PPG'] * player['season_GP'] +
                            player['season_APG'] * player['season_GP'] * 2.33)
        rebounds_generated = player['season_RBG'] * player['season_GP'] * 1.15
        defensive_contribution = (player['season_SPG'] * player['season_GP'] * 2.25 +
                                  player['season_BPG'] * player['season_GP'] * 2)
        turnovers_loss = player['season_TPG'] * player['season_GP'] * 2.25

        net_contribution = points_generated + rebounds_generated + defensive_contribution - turnovers_loss

        if player_merit_sum < 5:
            player_merit_sum = 5

        team_stats.at[index, 'player_merit'] = net_contribution * math.log(player_merit_sum)

    # --- 2. Sort by merit ---
    team_stats.sort_values(by='player_merit', ascending=False, inplace=True)

    # --- 3. Select best starting 5 formation ---
    starters = choose_best_formation(team_stats)
    if starters is None:
        starters = team_stats.head(5)

    # --- 4. Assign starter/bench roles ---
    team_stats['role'] = 'Bench'
    team_stats.loc[starters.index, 'role'] = 'Starter'

    # --- 5. Initialize seconds column ---
    team_stats['seconds'] = 0.0
    total_game_seconds = 240 * 60  # 14,400 seconds

    # Starter allocation
    starter_seconds_total = 160 * 60  # 9,600 seconds for 5 starters
    min_starter_seconds = 25 * 60
    max_starter_seconds = 42 * 60

    starters_seconds = starters['player_merit'] / starters['player_merit'].sum() * starter_seconds_total
    starters_seconds = starters_seconds.clip(lower=min_starter_seconds, upper=max_starter_seconds)

    # Rescale back to exactly starter_seconds_total
    starters_seconds *= starter_seconds_total / starters_seconds.sum()
    team_stats.loc[starters.index, 'seconds'] = starters_seconds

    # Bench allocation (top 5)
    bench_seconds_total = total_game_seconds - team_stats['seconds'].sum()
    bench = team_stats[team_stats['role'] == 'Bench']
    top5_bench = bench.nlargest(5, 'player_merit')

    if not top5_bench.empty and bench_seconds_total > 0:
        bench_merit_sum = top5_bench['player_merit'].sum()
        if bench_merit_sum > 0:
            bench_seconds = top5_bench['player_merit'] / bench_merit_sum * bench_seconds_total
        else:
            bench_seconds = [bench_seconds_total / len(top5_bench)] * len(top5_bench)
        team_stats.loc[top5_bench.index, 'seconds'] = bench_seconds

    # --- 6. Final adjustments ---
    # Clip to reasonable max for bench
    team_stats['seconds'] = team_stats['seconds'].clip(lower=0, upper=32*60)

    # Rescale total seconds to exactly total_game_seconds
    total_seconds = team_stats['seconds'].sum()
    if total_seconds > 0:
        team_stats['seconds'] = team_stats['seconds'] * (total_game_seconds / total_seconds)

    # Round to nearest second
    team_stats['seconds'] = team_stats['seconds'].round().astype(int)

    # --- 7. Remove last 5 unused players ---
    team_stats = team_stats[team_stats['seconds'] > 0].copy()

    return team_stats


class PlayerBoxScore:
    def __init__(self,Player_id,Player_name):
        self.name = Player_name
        self.id = Player_id
        self.points = 0
        self.ast = 0
        self.rb = 0
        self.stl = 0
        self.blk = 0
        #turnovers
        self.tov = 0
        #fouls
        self.foul = 0
        #field goals
        self.fgm = 0
        self.fga = 0
        self.fgp = 0.000
        #three point goals
        self.tpm = 0
        self.tpa = 0
        self.tpp = 0.000
        #free throws
        self.ftm = 0
        self.fta = 0
        self.ftp = 0.000
        #plus minus
        self.plus_minus = 0

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
        self.FG3M = player_df['season_FG3M']
        self.FG3A = player_df['season_FG3A']
        self.FG3_PCT = player_df['season_FG3_PCT']
        self.FTM = player_df['season_FTM']
        self.FTA = player_df['season_FTA']
        self.FT_PCT = player_df['season_FT_PCT']
        self.role = player_df['role']
        self.position = player_df['Position']
        
        #calculate tendencies
        '''
        self.two_point_tendency = (self.FGM - self.FG3M) / self.FGA if self.FGA > 0 else 0
        self.three_point_tendency = self.FG3M / self.FGA if self.FGA > 0 else 0

        ''' 
        self.total_seconds = 0 #total seconds played in game
        self.stint_seconds = 0 #seconds played in stint
        self.box = PlayerBoxScore(self.id,self.name)

        self.seconds = player_df['seconds']


        


class Team:
    def __init__(self,team_df):
        # Create Player objects
        self.players = {row['player_id']: Player(row) for _, row in team_df.iterrows()}
        self.starters = [p for p in self.players.values() if p.role == 'Starter']
        self.original_starters = [p for p in self.players.values() if p.role == 'Starter']
        self.bench = [p for p in self.players.values() if p.role == 'Bench']
        self.wins = 0
        self.losses = 0

class Game:
    def __init__(self,team1,team2,minutes):
        self.team1 = team1
        self.team2 = team2

        self.time_left = 60 * 48 #48 minutes * 60 secs/min
        self.timer = 0

        self.team1_score = 0
        self.team2_score = 0
        self.team1_total_timeouts = 7
        self.team2_total_timeouts = 7
        self.curr_possession = None
        self.current_quarter = 1
        self.overtime = False

        #possession counter for both teams
        self.poss_counter1 = 0
        self.poss_counter2 = 0
        self.last_sub_poss1 = -10
        self.last_sub_poss2 = -10

        #all occurances within the game
        self.occurances = []
    
    def tip_off(self):
        r = random.randint(1,2)
        self.curr_possession = 'Team1' if r == 1 else 'Team2'

    def end_game(self):
        if self.time_left == 0:
            if self.team1_score == self.team2_score:
                #set up overtime
                self.overtime = True
                self.current_quarter = 1

    def simulate_full_game(self):
            """
            Simulate a full NBA game (4 quarters, plus overtime if needed).
            Each possession uses self.simulate_possession.
            """

            def log(msg):
                print(msg)
                self.occurances.append(msg)

            
            self.current_quarter = 1

            while self.current_quarter <= 4 or (self.overtime and self.time_left > 0):
                log(f"=== Quarter {self.current_quarter} ===")
                self.time_left = 12 * 60  # 1 quarter = 12 * 60 seconds

                while self.time_left > 0:
                    # Simulate a possession
                    t = self.simulate_possession(self.time_left)
                    t = max(0,min(self.time_left,t))
                    self.time_left -= t
                    # Approximate possession duration (could sum actual action times for precision)
                    #self.time_left -= 15  

                log(f"Quarter {self.current_quarter} Score: Team1 {self.team1_score} - Team2 {self.team2_score}\n")
                self.current_quarter += 1

            # Check for overtime if tied
            if self.team1_score == self.team2_score:
                log("Game tied! Starting overtime...")
                self.overtime = True
                self.current_quarter = 1
                self.time_left = 12 * 60  # 5-minute overtime
                while self.time_left > 0:
                    t = self.simulate_possession(self.time_left)
                    t = max(0,min(self.time_left,t))
                    self.time_left -= t
                    

            log("=== Final Score ===")
            log(f"Team1: {self.team1_score}")
            log(f"Team2: {self.team2_score}\n")


            #debug seconds
            total_game_seconds = 4 * 12 * 60  # regulation seconds
            expected_team_player_seconds = total_game_seconds * 5
            team1_total = sum(p.total_seconds for p in self.team1.starters + self.team1.bench)
            team2_total = sum(p.total_seconds for p in self.team2.starters + self.team2.bench)
            log(f"[DEBUG] expected seconds per team (5 players * game seconds): {expected_team_player_seconds}")
            log(f"[DEBUG] actual Team1 summed player seconds: {team1_total}")
            log(f"[DEBUG] actual Team2 summed player seconds: {team2_total}")

            # Print box scores
            def print_box(team, name):
                log(f"--- {name} Box Score ---")
                total_secs = 0
                total_shot_attempts = 0
                for p in team.starters + team.bench:
                    rec_mins = p.seconds // 60
                    rec_secs = p.seconds % 60


                    log(f"{p.name}'s recommended minutes: {rec_mins}:{rec_secs:02d}")
                    total_secs += p.total_seconds
                    total_shot_attempts += p.box.fga
                    #convert total seconds into minutes : seconds
                    minutes = p.total_seconds // 60
                    seconds = p.total_seconds % 60

                    b = getattr(p, "box", None)
                    if b:
                        log(f"{p.name}: PTS {b.points}, AST {b.ast}, TOV {b.tov}, "
                            f"FGM/FGA {b.fgm}/{b.fga}, 3PM/3PA {b.tpm}/{b.tpa}, "
                            f"STL {b.stl}, BLK {b.blk}, Time {minutes}:{seconds:02d}"
                            f",plus/minus: {b.plus_minus}")
                log("\n")
                print(f"team total seconds {total_secs}\n")
                print(f"shots attempted: {total_shot_attempts}\n")

            print_box(self.team1, "Team1")
            print_box(self.team2, "Team2")

            with open("game_log.txt", "w", encoding="utf-8") as f:
                for line in self.occurances:
                    f.write(line + "\n")
        
    def quarter(self):
        self.time_left = 12 * 60 #12 minutes * 60 secs/min
        if self.current_quarter == 4:
            self.quarter_team1_timeouts = 3
            self.quarter_team2_timeouts = 3
        else:
            self.quarter_team1_timeouts = 2
            self.quarter_team2_timeouts = 2

    def perform_substitution(self,team, actions,players_on_court,total_game_seconds=48*60):
        """
        Substitutes players considering:
        - Starter cap (85% of total game time)
        - Plus-minus
        - Recommended seconds
        - Original starter priority
        """
        MIN_STINT = 60 * 6
        STARTER_CAP = total_game_seconds * 4 / 5  # 85% of game time

        # --- Force substitution for any starter exceeding the cap ---
        over_cap_starters = [s for s in team.starters if s.total_seconds >= STARTER_CAP]
        if over_cap_starters:
            sub_out = max(over_cap_starters, key=lambda p: p.total_seconds)
        else:
            # Eligible starters by stint or recommended seconds
            eligible = [
                s for s in team.starters
                if s.stint_seconds >= MIN_STINT or s.total_seconds >= getattr(s, "seconds", 0)
            ]
            if not eligible:
                return  # no sub needed
            # Pick starter with lowest plus-minus among eligible
            sub_out = min(eligible, key=lambda p: p.box.plus_minus)

        # --- Select sub-in player ---
        bench_candidates = [b for b in team.bench if b not in team.starters]
        # Prefer original starters on bench
        original_on_bench = [b for b in getattr(team, "original_starters", []) if b in bench_candidates]
        if original_on_bench:
            sub_in = min(original_on_bench, key=lambda b: b.total_seconds)
            candidates = original_on_bench
        else:
            same_pos_bench = [b for b in bench_candidates if b.position == sub_out.position]
            sub_in = min(same_pos_bench, key=lambda b: b.total_seconds) if same_pos_bench else min(bench_candidates, key=lambda b: b.total_seconds)
            candidates = bench_candidates
        
        
        # --- Weighted selection based on remaining allowed minutes ---
        remaining_allowed = [max(getattr(b, 'seconds', 0) - b.total_seconds, 1) for b in candidates]
        total_remaining = sum(remaining_allowed)
        weights = [r / total_remaining for r in remaining_allowed]

        sub_in = random.choices(candidates, weights=weights, k=1)[0]
        # --- Swap players ---
        idx_starter = team.starters.index(sub_out)
        idx_bench = team.bench.index(sub_in)
        team.starters[idx_starter], team.bench[idx_bench] = sub_in, sub_out

        # --- Reset stint timers ---
        sub_out.stint_seconds = 0
        sub_in.stint_seconds = 0

        # --- Update players on court ---
        players_on_court = team.starters

        # --- Log substitution ---
        actions += (
            f"Substitution: {sub_out.name} out, {sub_in.name} in "
            f"(plus-minus: {sub_out.box.plus_minus}, total seconds: {sub_out.total_seconds}/{getattr(sub_out,'seconds',0)})\n"
        )

    def simulate_possession(self, remaining_secs):
        """
        Simulate a full basketball possession for the current team:
        - 24-second shot clock
        - Passes, assists, turnovers tracked
        - Steals and blocks tracked with defenders' SPG/BPG weights
        - 2-point vs 3-point shot accuracy
        - Last-second shot if quarter ends
        - Substitutions for starters based on seconds played
        """

        team = self.team1 if self.curr_possession == 'Team1' else self.team2
        defense = self.team2 if self.curr_possession == 'Team1' else self.team1

        if self.curr_possession == 'Team1':
            self.poss_counter1 += 1
        else:
            self.poss_counter2 += 1

        ball_handler = max(team.starters, key=lambda p: p.apg) if team.starters else random.choice(list(team.players.values()))
        players_on_court = team.starters
        defenders_on_court = defense.starters

        shot_clock = 24
        possession_time = 0
        actions = f"{ball_handler.name} starts with the ball\n"
        passes = 0
        last_passer = None
        points = 0
        result = None

        rem = int(remaining_secs)

        while shot_clock > 0 and rem > 0:
            # Action time
            action_time = random.choices([1,2,3,4,5,6,7,8], weights=[0.05,0.1,0.15,0.2,0.1,0.1,0.1,0.1], k=1)[0]
            action_time = min(action_time, shot_clock, rem)
            last_shot = shot_clock <= action_time or rem <= action_time

            shot_clock -= action_time
            rem -= action_time
            possession_time += action_time

            # Update defenders
            defenders_on_court = defense.starters

            # --- Calculate probabilities ---
            turnover_prob = min(ball_handler.tpg / max(ball_handler.gp,1) + 0.02, 0.1)
            shooting_volume = ball_handler.FGM / max(ball_handler.gp,1)
            shooting_efficiency = ball_handler.FGM / max(ball_handler.FGA,1) if ball_handler.FGA else 0.45

            # Low-volume boost for efficient shooters
            volume_f = min(shooting_volume / 20, 1)
            low_volume_boost = (1 - volume_f) * shooting_efficiency * 0.3

            score_weight = shooting_efficiency + low_volume_boost
            assist_weight = ball_handler.apg / max(ball_handler.gp,1)

            # Normalize weights to probabilities
            total_offense = score_weight + assist_weight
            shot_prob = (score_weight / total_offense) * (1 - turnover_prob)
            pass_prob = (assist_weight / total_offense) * (1 - turnover_prob)
            dribble_prob = max(0.05, 1 - (shot_prob + pass_prob + turnover_prob))

            # Final normalization
            sum_probs = pass_prob + shot_prob + dribble_prob + turnover_prob
            pass_prob /= sum_probs
            shot_prob /= sum_probs
            dribble_prob /= sum_probs
            turnover_prob /= sum_probs

            # --- DEBUG: Print probabilities ---
            print(f"{ball_handler.name} action probabilities:")
            print(f"  Pass: {pass_prob:.3f}")
            print(f"  Dribble: {dribble_prob:.3f}")
            print(f"  Shot: {shot_prob:.3f}")
            print(f"  Turnover: {turnover_prob:.3f}")

            # --- Choose action ---
            action = random.choices(
                ["pass","dribble","shot","turnover"],
                weights=[pass_prob, dribble_prob, shot_prob, turnover_prob],
                k=1
            )[0]

            # --- PASS ACTION ---
            if action == "pass":
                steal_weights = [p.spg / max(p.gp, 1) + 0.01 for p in defenders_on_court]
                if random.random() < 0.03:
                    stealer = random.choices(defenders_on_court, weights=steal_weights, k=1)[0]
                    stealer.box.stl += 1
                    ball_handler.box.tov += 1
                    actions += f"{stealer.name} steals the ball from {ball_handler.name}!\n"
                    result = "steal"
                    self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                    break

                passes += 1
                last_passer = ball_handler
                new_handler = random.choice([p for p in players_on_court if p != ball_handler])
                actions += f"{ball_handler.name} passes to {new_handler.name} ({action_time}s)\n"
                ball_handler = new_handler

            # --- DRIBBLE ACTION ---
            elif action == "dribble":
                steal_weights = [p.spg / max(p.gp, 1) + 0.01 for p in defenders_on_court]
                if random.random() < 0.03:
                    stealer = random.choices(defenders_on_court, weights=steal_weights, k=1)[0]
                    stealer.box.stl += 1
                    ball_handler.box.tov += 1
                    actions += f"{stealer.name} steals the ball from {ball_handler.name} while dribbling!\n"
                    result = "steal"
                    self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                    break
                actions += f"{ball_handler.name} dribbles ({action_time}s)\n"

            # --- SHOT ACTION ---
            elif action == "shot" or last_shot:
                shooter = ball_handler
                three_point_rate = shooter.FG3A / max(shooter.FGA,1) if shooter.FGA else 0.1
                is_three = random.random() < three_point_rate
                points = 3 if is_three else 2
                fg_chance = shooter.FG3_PCT if is_three and shooter.FG3_PCT else shooter.FG_PCT
                fg_chance = fg_chance if fg_chance else 0.45

                block_weights = [p.bpg / max(p.gp, 1) + 0.01 for p in defenders_on_court]
                if random.random() < 0.05:
                    points = 0
                    blocker = random.choices(defenders_on_court, weights=block_weights, k=1)[0]
                    blocker.box.blk += 1
                    shooter.box.fga += 1
                    actions += f"{blocker.name} blocks the shot of {shooter.name}!\n"
                    result = "block"
                    self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                    break

                made = random.random() < fg_chance
                if made:
                    actions += f"{shooter.name} makes a {points}-point shot! ({action_time}s)\n"
                    result = "score"
                    shooter.box.fgm += 1
                    shooter.box.fga += 1
                    shooter.box.points += points
                    if is_three:
                        shooter.box.tpm += 1
                        shooter.box.tpa += 1
                        shooter.box.tpp = round(shooter.box.tpm / shooter.box.tpa,3)
                    if passes > 0 and last_passer and last_passer != shooter:
                        last_passer.box.ast += 1
                else:
                    actions += f"{shooter.name} misses a {points}-point shot ({action_time}s)\n"
                    shooter.box.fga += 1
                    if is_three:
                        shooter.box.tpa += 1
                        shooter.box.tpp = round(shooter.box.tpm / shooter.box.tpa,3)
                    points = 0
                shooter.box.fgp = round(shooter.box.fgm / max(shooter.box.fga,1), 3)
                self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                break

            # --- TURNOVER ACTION ---
            elif action == "turnover":
                actions += f"{ball_handler.name} turns the ball over ({action_time}s)\n"
                ball_handler.box.tov += 1
                result = "turnover"
                self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                break

        # --- Update scores and plus/minus ---
        if points > 0:
            if self.curr_possession == 'Team2':
                self.team1_score += points
            else:
                self.team2_score += points

            for p in defenders_on_court:
                p.box.plus_minus -= points
            for p in players_on_court:
                p.box.plus_minus += points

        # --- Update timers ---
        for p in players_on_court + defenders_on_court:
            p.total_seconds = getattr(p, "total_seconds", 0) + possession_time
            p.stint_seconds = getattr(p, "stint_seconds", 0) + possession_time

        # --- Perform substitutions ---
        MIN_POSSESSIONS_BETWEEN_SUBS = 3
        if (self.poss_counter1 - self.last_sub_poss1) >= MIN_POSSESSIONS_BETWEEN_SUBS:
            self.perform_substitution(self.team1,actions,self.team1.starters)
            self.last_sub_poss1 = self.poss_counter1
        if (self.poss_counter2 - self.last_sub_poss2) >= MIN_POSSESSIONS_BETWEEN_SUBS:
            self.perform_substitution(self.team2,actions,self.team2.starters)
            self.last_sub_poss2 = self.poss_counter2

        actions += f"=== Current Score ===\nTeam1: {self.team1_score} Team2: {self.team2_score}\n"
        self.occurances.append(actions)

        return possession_time




if __name__ == "__main__":
    cavs_2016 = [
        "LeBron James", "Kyrie Irving", "Kevin Love", "Tristan Thompson", "J.R. Smith",
        "Richard Jefferson", "Channing Frye", "Matthew Dellavedova",
        # add 7 more role players from the 2016 Cavs or random fill
        "Iman Shumpert", "Mo Williams", "Timofey Mozgov", "James Jones", "Sasha Kaun",
        "Jordan McRae", "Dahntay Jones"
    ]

    team1 = make_team(cavs_2016)   # Cavs 2016 roster

    warriors_2016 = [
            "Stephen Curry", "Klay Thompson", "Draymond Green", "Harrison Barnes", "Andrew Bogut",
            "Andre Iguodala", "Shaun Livingston", "Festus Ezeli", "Marreese Speights",
            "Leandro Barbosa", "Brandon Rush", "James Michael McAdoo", "Ian Clark", "Matt Barnes","Aaron Brooks"
        ]



    team2 = make_team(warriors_2016) 
    #print(team1)
    #print(team2)
    sorted_team1 = determine_seconds(team1)
    sorted_team2 = determine_seconds(team2)
    #adjust minutes for team1 and 2


    #print(sorted_team1[['full_name', 'Position', 'role','player_merit', 'minutes']])
    #print("\n")
    #print("minutes total = ",sorted_team1['minutes'].sum())
    #print(sorted_team2[['full_name', 'Position', 'role','player_merit', 'minutes']])
    #print("minutes total = ",sorted_team2['minutes'].sum())

    Team1 = Team(sorted_team1)
    Team2 = Team(sorted_team2)

    #print(Team1.starters[0].gp)

    G = Game(Team1,Team2,48)
    G.tip_off()
    G.simulate_full_game()
    #G.simulate_possession()


    #simulate_game(sorted_team1, sorted_team2,48)



        

   




