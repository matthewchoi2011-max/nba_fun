import json
import pandas as pd
import random
import math
import os
import json
import numpy as np
import unicodedata
from difflib import get_close_matches

class PlayerBoxScore:
    def __init__(self,Player_name):
        self.name = Player_name
        self.points = 0
        self.ast = 0
        self.oreb = 0
        self.dreb = 0
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
    def __init__(self, player, player_data):
        self.name = player
        stats = player_data.get('stats', {})
        awards = player_data.get('awards', {})

        # --- Stats ---
        self.ppg = stats.get('PPG', 0)
        self.apg = stats.get('APG', 0)
        self.rpg = stats.get('RBG', 0)
        self.spg = stats.get('SPG', 0)
        self.bpg = stats.get('BPG', 0)
        self.tpg = stats.get('TPG', 0)
        self.gp = stats.get('total_games', 0)
        self.FGM = stats.get('FGM', 0)
        self.FGA = stats.get('FGA', 0)
        self.FG_PCT = stats.get('FG_PCT', 0)
        self.FG3M = stats.get('FG3M', 0)
        self.FG3A = stats.get('FG3A', 0)
        self.FG3_PCT = stats.get('FG3_PCT', 0)
        self.FTM = stats.get('FTM', 0)
        self.FTA = stats.get('FTA', 0)
        self.FT_PCT = stats.get('FT_PCT', 0)


        # --- Attributes ---
        self.position = player_data.get('Position', 'Unknown')
        self.role = player_data.get('role', 'Bench')
        self.height = player_data.get('height', 0)
        self.weight = player_data.get('weight', 0)
        self.merit = 0

        # --- Timing ---
        self.total_seconds = 0
        self.recommended_seconds = player_data.get('seconds', 0)
        self.game_recommended_seconds = 0
        self.stint_seconds = 0
        self.bench_stint_seconds = 0

        self.box = PlayerBoxScore(self.name)


class Team:
    def __init__(self, team_dict):

        self.players = {pid: Player(pid, pdata) for pid, pdata in team_dict.items()}
        self.starters = [p for p in self.players.values() if p.role == 'Starter']
        self.original_starters = self.starters
        self.bench = [p for p in self.players.values() if p.role == 'Bench']
        self.original_bench = self.bench

        self.name = ""
        self.schedule = ""




def choose_best_formation(team: Team):
    """
    Select the best starting 5 from a Team object based on player merit.
    Returns a list of Player objects for starters.
    """

    # Helper to select players for a formation
    def try_formation(requirements):
        chosen = []
        remaining = list(team.players.values())
        for pos, count in requirements.items():
            # Filter players by position
            candidates = [p for p in remaining if pos in p.position.upper()]
            # Sort by merit
            candidates.sort(key=lambda x: x.merit, reverse=True)
            if len(candidates) < count:
                return None  # formation not possible
            # Take top `count` players
            top_players = candidates[:count]
            chosen.extend(top_players)
            # Remove chosen from remaining
            for p in top_players:
                remaining.remove(p)
        return chosen

    # Define formations
    formations = [
        {"G": 3, "F": 2},          # G G G F F
        {"G": 2, "F": 2, "C": 1},  # G G F F C
        {"G": 1, "F": 3, "C": 1},  # G F F F C
        {"G": 2, "F": 3}           # G G F F F
    ]

    best = None
    best_merit = -1
    for formation in formations:
        lineup = try_formation(formation)
        if lineup is not None:
            merit_sum = sum(p.merit for p in lineup)
            if merit_sum > best_merit:
                best_merit = merit_sum
                best = lineup

    # Update team starters and bench
    if best:
        team.starters = best
        team.bench = [p for p in team.players.values() if p not in best]

    

def determine_seconds_and_make_team(team_data, top_bench_boost=4):
    import math

    # --- Initialize team ---
    team = Team(team_data)

    # --- Calculate player merit ---
    for player_name, player_info in team_data.items():
        awards = player_info.get('awards', {})
        stats = player_info.get('stats', {})

        award_merit_sum = (
                           awards.get('NBA All-Star', 0) * 2.5 +
                           awards.get('All-NBA-First', 0) * 5 +
                           awards.get('All-NBA-Second', 0) * 4 +
                           awards.get('All-NBA-Third', 0) * 3 +
                           awards.get('Championships', 0) * 2 +
                           awards.get('MVP', 0) * 8 +
                           awards.get('FMVP', 0) * 4 +
                           awards.get('DPOY', 0) * 3.5 +
                           awards.get('All-Defensive-First', 0) * 1.8 +
                           awards.get('All-Defensive-Second', 0) * 1.3)
        
        if award_merit_sum == 0 and stats.get('total_games') == 0:
            award_merit_sum = random.randint(1500,3500) #determine merit for Rookies! 

        points_contributed = stats.get('PPG', 0) * stats.get('total_games', 0)
        assists_contributed = stats.get('APG', 0) * stats.get('total_games', 0)
        if stats.get('FG3_PCT', 0) > 0.38 and stats.get('FG3M', 0) > 100:
            points_contributed *= 1.5
        offensive_merit = math.sqrt(points_contributed * assists_contributed)

        steals_generated = stats.get('SPG', 0) * stats.get('total_games', 0)
        blocks_generated = stats.get('BPG', 0) * stats.get('total_games', 0)
        turnovers_generated = stats.get('TPG', 0) * stats.get('total_games', 0)
        defensive_merit = math.sqrt(steals_generated * blocks_generated) - turnovers_generated / 10

        merit = int(award_merit_sum * 0.25 + offensive_merit * 0.55 + defensive_merit * 0.45)
        merit = max(merit, 100)

        if player_name in team.players:
            team.players[player_name].merit = merit

    # --- Choose starters ---
    choose_best_formation(team)

    # --- Assign roles ---
    for player in team.players.values():
        player.role = "Starter" if player in team.starters else "Bench"

    # --- Parameters ---
    min_starter_seconds = 28 * 60
    max_starter_seconds = 40 * 60
    min_bench_seconds = 0 * 60
    max_bench_seconds = 30 * 60
    target_starters_total = 150 * 60  # 5 starters
    target_bench_total = 90 * 60      # bench total

    # --- Nonlinear merit-to-seconds scaling ---
    def merit_to_seconds(merit, base_merit, base_seconds, max_seconds):
        if merit <= base_merit:
            return base_seconds
        scaled = base_seconds + math.sqrt(max(0, merit - base_merit)) * 0.4 * 60
        return min(max(scaled, base_seconds), max_seconds)

    # --- Initial recommended seconds ---
    for p in team.starters:
        p.game_recommended_seconds = merit_to_seconds(p.merit, 400, min_starter_seconds, max_starter_seconds)
    for p in team.bench:
        p.game_recommended_seconds = merit_to_seconds(p.merit, 60, min_bench_seconds, max_bench_seconds)

    # --- Scale starters proportionally ---
    starters = list(team.starters)
    starter_sum = sum(p.game_recommended_seconds for p in starters)
    starter_scale = target_starters_total / starter_sum if starter_sum > 0 else 1.0
    for p in starters:
        p.game_recommended_seconds = min(max(p.game_recommended_seconds * starter_scale, min_starter_seconds), max_starter_seconds)

    # --- Scale bench proportionally, top N chosen by merit ---
    bench = list(team.bench)
    if bench:
        # Sort bench by merit descending
        bench_sorted = sorted(bench, key=lambda x: x.merit, reverse=True)
        top_n = min(top_bench_boost, len(bench_sorted))

        # Boost top N bench players slightly
        for i, p in enumerate(bench_sorted[:top_n]):
            p.game_recommended_seconds *= 1.1  # 10% boost

        # Scale all bench proportionally to fit target_bench_total
        bench_sum_after_boost = sum(p.game_recommended_seconds for p in bench_sorted)
        scale_factor = target_bench_total / bench_sum_after_boost if bench_sum_after_boost > 0 else 1.0
        for p in bench_sorted:
            p.game_recommended_seconds = min(max(p.game_recommended_seconds * scale_factor, min_bench_seconds), max_bench_seconds)

    # --- Final normalization for all players to match 48*5 minutes total ---
    all_players = starters + bench
    total_after = sum(p.game_recommended_seconds for p in all_players)
    final_factor = (48 * 60 * 5) / total_after if total_after > 0 else 1.0
    for p in all_players:
        p.game_recommended_seconds = round(p.game_recommended_seconds * final_factor, 1)
    
    team.original_starters = list(team.starters)
    team.original_bench = list(team.bench)

    return team





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

        self.winner = None
    
    def tip_off(self):
        r = random.randint(1,2)
        self.curr_possession = 'Team1' if r == 1 else 'Team2'

    def end_game(self):
        if self.time_left == 0:
            if self.team1_score == self.team2_score:
                # Set up overtime
                self.overtime = True
                self.current_quarter = 1  # or set to an overtime-specific value
            else:
                self.overtime = False
                self.winner = 'Team1' if self.team1_score > self.team2_score else 'Team2'


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
                t = self.simulate_possession(self.time_left,24)
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
                t = self.simulate_possession(self.time_left,24)
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
            log(f"=== {name} Box Score ===\n")

            def print_players(players, label):
                log(f"--- {label} ---")
                total_secs = 0
                total_shot_attempts = 0
                total_shots_made = 0

                for p in players:
                    rec_mins = int(p.game_recommended_seconds) // 60
                    rec_secs = int(p.game_recommended_seconds) % 60
                    log(f"{p.name}'s recommended minutes: {rec_mins}:{rec_secs:02d}")

                    total_secs += p.total_seconds
                    total_shots_made += p.box.fgm
                    total_shot_attempts += p.box.fga

                    minutes = p.total_seconds // 60
                    seconds = p.total_seconds % 60
                    b = getattr(p, "box", None)
                    if b:
                        log(f"{p.name}, {p.position}: PTS {b.points}, AST {b.ast}, REB {b.oreb + b.dreb}, "
                            f"TOV {b.tov}, FGM/FGA {b.fgm}/{b.fga}, 3PM/3PA {b.tpm}/{b.tpa}, "
                            f"FTM/FTA {b.ftm}/{b.fta}, STL {b.stl}, BLK {b.blk}, "
                            f"Time {minutes}:{seconds:02d}, Plus/Minus {b.plus_minus}, Fouls {b.foul}")

                log(f"{label} total seconds: {total_secs}")
                log(f"{label} shots made: {total_shots_made}, shots attempted: {total_shot_attempts}\n")

            # Print starters first
            print_players(team.original_starters, "Starters")

            # Then print bench
            print_players(team.original_bench, "Bench")

            log("\n")

        print_box(self.team1, "Team1")
        print_box(self.team2, "Team2")

        with open("game_log.txt", "w", encoding="utf-8") as f:
            for line in self.occurances:
                f.write(line + "\n")
        
        if self.team1_score > self.team2_score:
            return "Team1"
        else:
            return "Team2"
        
    def quarter(self):
        self.time_left = 12 * 60 #12 minutes * 60 secs/min
        if self.current_quarter == 4:
            self.quarter_team1_timeouts = 3
            self.quarter_team2_timeouts = 3
        else:
            self.quarter_team1_timeouts = 2
            self.quarter_team2_timeouts = 2

    def perform_substitution(self, team, actions, players_on_court, original_starters):
        """
        Substitution with fatigue awareness and flexible positions:
        - Avoid excessive consecutive minutes.
        - Ensure realistic playing time.
        - Always sub out fouled-out or overplayed players.
        - Flexible position rules (e.g., C can replace F, F can replace C).
        """
        MAX_GAME_SECONDS = 48 * 60
        MIN_STINT = 60 * 3
        MAX_STINT = 60 * random.randint(6, 16)
        MIN_REST = 60 * 2.5
        MAX_FOULS = 5

        def get_compatible_bench(sub_out, bench_list):
            compatible = []
            for b in bench_list:
                # Exact position match
                if b.position[0].lower() == sub_out.position[0].lower():
                    compatible.append(b)
                # Flexibility: C <-> F
                elif sub_out.position[0].lower() in ["c","f"] and b.position[0].lower() in ["c","f"]:
                    compatible.append(b)
                # Optionally extend: G <-> SG/PG, etc.
            return compatible

        # --- Mandatory subs ---
        mandatory_subs = [
            s for s in team.starters
            if s.box.foul >= 6 or s.total_seconds >= s.game_recommended_seconds * random.uniform(0.9, 1.1)
        ]

        for sub_out in mandatory_subs:
            bench_candidates = [
                b for b in team.bench
                if b.total_seconds < MAX_GAME_SECONDS and b.box.foul < MAX_FOULS
            ]
            if not bench_candidates:
                continue

            compatible_bench = get_compatible_bench(sub_out, bench_candidates)
            if compatible_bench:
                chosen_pool = compatible_bench
                actions += "Compatible Replacement Bench Found!\n"
            else:
                chosen_pool = bench_candidates
                actions += "Compatible Replacement Bench Not Found!\n"

            # Weighted selection based on remaining recommended seconds
            remaining_needed = [max(b.recommended_seconds - b.total_seconds, 1) for b in chosen_pool]
            total_needed = sum(remaining_needed)
            weights = [r / total_needed for r in remaining_needed]
            sub_in = random.choices(chosen_pool, weights=weights, k=1)[0]

            # Swap players
            idx_starter = team.starters.index(sub_out)
            idx_bench = team.bench.index(sub_in)
            team.starters[idx_starter], team.bench[idx_bench] = sub_in, sub_out
            players_on_court[:] = team.starters

            actions += (
                f"Forced Substitution: {sub_out.name} OUT ({sub_out.box.foul} fouls, "
                f"{sub_out.stint_seconds//60:.1f} min) — {sub_in.name} IN\n"
            )

            # Reset stint timers
            sub_out.stint_seconds = 0
            sub_in.stint_seconds = 0
            sub_in.bench_stint_seconds = 0

        if mandatory_subs:
            return actions

        # --- Optional subs (fatigue/foul) ---
        optional_subs = []
        for s in team.starters:
            fatigue = (s.stint_seconds / MAX_STINT) * 0.7 + (s.total_seconds / s.game_recommended_seconds) * 0.3
            if s.box.foul >= 5 or s.stint_seconds >= MAX_STINT or (fatigue > 1.0 and s.stint_seconds >= MIN_STINT):
                optional_subs.append(s)

        if not optional_subs:
            return actions

        sub_out = max(optional_subs, key=lambda s: s.stint_seconds / MAX_STINT + s.total_seconds / s.game_recommended_seconds)

        bench_candidates = [
            b for b in team.bench
            if b.bench_stint_seconds >= MIN_REST and b.total_seconds < MAX_GAME_SECONDS and b.box.foul < MAX_FOULS
        ]
        if not bench_candidates:
            return actions

        compatible_bench = get_compatible_bench(sub_out, bench_candidates)
        chosen_pool = compatible_bench if compatible_bench else bench_candidates
        if compatible_bench:
            actions += "Compatible Replacement Bench Found!\n"
        else:
            actions += "Compatible Replacement Bench Not Found!\n"

        remaining_needed = [max(b.recommended_seconds - b.total_seconds, 1) for b in chosen_pool]
        total_needed = sum(remaining_needed)
        weights = [r / total_needed for r in remaining_needed]
        sub_in = random.choices(chosen_pool, weights=weights, k=1)[0]

        # Swap players
        idx_starter = team.starters.index(sub_out)
        idx_bench = team.bench.index(sub_in)
        team.starters[idx_starter], team.bench[idx_bench] = sub_in, sub_out
        players_on_court[:] = team.starters

        actions += (
            f"Substitution: {sub_out.name} OUT ({sub_out.stint_seconds//60:.1f} min) — "
            f"{sub_in.name} IN (rested {sub_in.bench_stint_seconds//60:.1f} min)\n"
        )

        sub_out.stint_seconds = 0
        sub_in.stint_seconds = 0
        sub_in.bench_stint_seconds = 0

        return actions


    def free_throw(self,player,num,action):

        """
        Simulate Free throw attempts
        """
        for __ in range(num):

            if player.FT_PCT == 0:
                player.FT_PCT = random.uniform(0.6,0.85)
            if random.random() <= player.FT_PCT:
                player.box.ftm += 1
                player.box.points += 1
                action += f"{player.name} makes free throw\n"
                #add to scoreboard
                if player in self.team1.starters + self.team1.bench:
                    self.team1_score += 1
                else:
                    self.team2_score += 1

                #update box scores
                if player in self.team1.starters + self.team1.bench:                  
                    for p in self.team1.starters:
                        p.box.plus_minus += 1
                    for p in self.team2.starters:
                        p.box.plus_minus -= 1
                else:
                    for p in self.team1.starters:
                        p.box.plus_minus -= 1
                    for p in self.team2.starters:
                        p.box.plus_minus += 1
            else:
                action += f"{player.name} missed free throw\n"
            player.box.fta += 1
            action += (f"=== Current Score ===\nTeam1: {self.team1_score} Team2: {self.team2_score}\n")
        return action

    def simulate_possession(self, remaining_secs, shot_clock):
        """
        Simulate a full basketball possession for the current team (defensive, zero-denominator safe).
        """

        def safe_gp(p):
            return max(int(getattr(p, "gp", 0)) or 0, 1)

        def safe_attr(p, name, default=0.0):
            return getattr(p, name, default)

        team = self.team1 if self.curr_possession == 'Team1' else self.team2
        defense = self.team2 if self.curr_possession == 'Team1' else self.team1

        if self.curr_possession == 'Team1':
            self.poss_counter1 += 1
        else:
            self.poss_counter2 += 1

        # pull starters from team class (assume team.starters exists)
        players_on_court = list(getattr(team, "starters", []) or [])
        defenders_on_court = list(getattr(defense, "starters", []) or [])

        # If not enough, fill from team.players (dictionary or list)
        if len(players_on_court) < 5:
            pool = getattr(team, "players", None)
            pool_list = list(pool.values()) if isinstance(pool, dict) else (list(pool) if pool else [])
            while len(players_on_court) < 5:
                if not pool_list:
                    # Give a clear error so upper layers can handle it (or you can change to a fallback)
                    raise ValueError(f"No players available for {team.name} to fill starters.")
                players_on_court.append(random.choice(pool_list))

        if len(defenders_on_court) < 5:
            pool = getattr(defense, "players", None)
            pool_list = list(pool.values()) if isinstance(pool, dict) else (list(pool) if pool else [])
            while len(defenders_on_court) < 5:
                if not pool_list:
                    raise ValueError(f"No defenders available for {defense.name} to fill starters.")
                defenders_on_court.append(random.choice(pool_list))

        # prepare weights for ball handling (safe: avoid zero denominators)
        weights = []
        for p in players_on_court:
            gp = safe_gp(p)
            apg = safe_attr(p, "apg", 0.0)
            ppg = safe_attr(p, "ppg", 0.0)
            FGM = safe_attr(p, "FGM", 0.0)
            FGA = safe_attr(p, "FGA", 0.0)

            assist_weight = 1 + (apg * 1.2 / 5) ** 2.5 if apg >= 5 else 0.35
            base_weight = (assist_weight + ppg * 0.35 + max(0.3, FGM / gp))
            fga_per_game = (FGA / gp) if gp > 0 else 0
            if fga_per_game > 13.5 and apg >= 5:
                # keep the modifier bounded to avoid huge weights
                base_weight *= (FGA / (gp * 3))
            weights.append(max(base_weight, 0.01))  # ensure positive

        # safety for random.choices: weights must match population and sum > 0
        if len(weights) != len(players_on_court) or sum(weights) <= 0:
            # fallback uniform weights (log once)
            print(f"⚠️ Warning: bad pass-weights for {team.name}. Using uniform weights.")
            weights = [1] * len(players_on_court)

        # pick initial ball handler
        ball_handler = random.choices(players_on_court, weights=weights, k=1)[0]

        possession_time = 0
        actions = f"{getattr(ball_handler, 'name', str(ball_handler))} starts with the ball\n"
        passes = 0
        last_passer = None
        points = 0
        result = None

        rem = int(remaining_secs)

        # main possession loop
        while shot_clock > 0 and rem > 0:
            # action time safely chosen
            action_time = random.choices(
                [1,2,3,4,5,6,7,8],
                weights=[0.025,0.1,0.15,0.225,0.225,0.15,0.1,0.025],
                k=1
            )[0]
            action_time = min(action_time, shot_clock, rem)
            last_shot = shot_clock <= action_time or rem <= action_time

            shot_clock -= action_time
            rem -= action_time
            possession_time += action_time

            # refresh defenders on court from defense (in case of substitutions)
            defenders_on_court = list(getattr(defense, "starters", []) or defenders_on_court)

            # --- Calculate probabilities (safe arithmetic) ---
            gp_bh = safe_gp(ball_handler)
            turnover_prob = min(safe_attr(ball_handler, "tpg", 0.0) / gp_bh * 0.05 + 0.02, 0.04)

            bh_FGM = safe_attr(ball_handler, "FGM", 0.0)
            bh_FGA = safe_attr(ball_handler, "FGA", 0.0)
            shooting_efficiency = (bh_FGM / max(bh_FGA, 1)) if bh_FGA else random.uniform(0.35, 0.55)

            shooting_volume = min(bh_FGM / max(gp_bh, 1) / 10.0, 1)
            #print(shooting_volume)
            volume_f = min(2 * shooting_volume, 1)
            low_volume_boost = (1 - volume_f) * shooting_efficiency * 0.1

            score_weight = shooting_efficiency * volume_f + low_volume_boost
            assist_weight = min(safe_attr(ball_handler, "apg", 0.0) / max(gp_bh, 1) * 70, 0.8)

            total_offense = score_weight * 1.5 + assist_weight
            # protect against divide by zero
            if total_offense <= 0:
                total_offense = 1e-6

            shot_prob = (score_weight / total_offense) * (1 - turnover_prob) * 0.4
            shot_prob = min(max(shot_prob, 0.0), 0.6)

            time_factor = max(min(shot_clock, rem) / 24.0, 0.0)
            dribble_prob = max(min((0.035 * shot_clock), 0.22), 0.03) * time_factor

            pass_prob = (1 - shot_prob - turnover_prob - dribble_prob) * time_factor
            ratio = safe_attr(ball_handler, "apg", 0.0)/ max(safe_attr(ball_handler, "ppg", 0.0),0.001)
            pass_prob *= min((ratio) * 4, 4.0)
            if pass_prob == 0: pass_prob = random.uniform(0.2,0.4)

            if min(shot_clock, rem) <= 0:
                pass_prob = 0
                dribble_prob = 0

            # final normalization (avoid zero-sum)
            sum_probs = pass_prob + shot_prob + dribble_prob + turnover_prob
            if sum_probs <= 0:
                # fallback: equalize
                pass_prob = shot_prob = dribble_prob = turnover_prob = 0.25
                sum_probs = 1.0

            pass_prob /= sum_probs
            shot_prob /= sum_probs
            dribble_prob /= sum_probs
            turnover_prob /= sum_probs

            # debug prints (optional)
            print(f"{ball_handler.name} action probs - Pass:{pass_prob:.3f} Dribble:{dribble_prob:.3f} Shot:{shot_prob:.3f} TOV:{turnover_prob:.3f}")

            # choose action safely
            action = random.choices(
                ["pass","dribble","shot","turnover"],
                weights=[pass_prob, dribble_prob, shot_prob, turnover_prob],
                k=1
            )[0]

            free_foul = False

            # --- PASS ACTION ---
            if action == "pass":
                # compute steal weights safely
                steal_weights = []
                for p in defenders_on_court:
                    gp = safe_gp(p)
                    spg = safe_attr(p, "spg", 0.0)
                    steal_weights.append(max((spg ** 2) / gp + 0.01, 0.001))

                if not defenders_on_court or sum(steal_weights) <= 0:
                    # fallback: no steal possible
                    steal_possible = False
                else:
                    steal_possible = True

                if steal_possible and random.random() < 0.04:
                    stealer = random.choices(defenders_on_court, weights=steal_weights, k=1)[0]
                    stealer.box.stl += 1
                    ball_handler.box.tov += 1
                    actions += f"{getattr(stealer,'name',str(stealer))} steals the ball from {getattr(ball_handler,'name',str(ball_handler))}!\n"
                    result = "steal"
                    self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                    break

                # pick a teammate to pass to; if only one, pass back to same (rare)
                teammate_weights = []
                for p in players_on_court:
                    if p is not ball_handler:
                        if p.ppg == 0:
                            
                            teammate_weights.append(random.uniform(0.3,0.8))
                        else:

                            teammate_weights.append(p.apg/p.ppg)

                teammates = [p for p in players_on_court if p is not ball_handler]
                
                if not teammates:
                    # fallback: keep ball with handler
                    new_handler = ball_handler
                else:
                    new_handler = random.choices(teammates,weights=teammate_weights,k=1)[0]

                passes += 1
                last_passer = ball_handler
                actions += f"{getattr(ball_handler,'name',str(ball_handler))} passes to {getattr(new_handler,'name',str(new_handler))} ({action_time}s)\n"
                ball_handler = new_handler

            # --- DRIBBLE ACTION ---
            elif action == "dribble":
                # defensive foul checks
                for defender in defenders_on_court:
                    gp = safe_gp(defender)
                    foul_weight = safe_attr(ball_handler, "FTA", 0.0) / max(safe_gp(ball_handler), 1)
                    foul_tendency = 1 / (1 + math.exp(-0.6 * (foul_weight - 5)))
                    foul_factor = 0.02 + foul_tendency * 0.2

                    if random.random() < foul_factor * 0.2:
                        defender.box.foul = getattr(defender.box, "foul", 0) + 1
                        actions += f"{getattr(defender,'name',str(defender))} commits a foul on {getattr(ball_handler,'name',str(ball_handler))}!\n"
                        if defender.box.foul >= 6:
                            actions = self.perform_substitution(defense, actions, defense.starters, defense.original_starters)
                            actions += f"{getattr(defender,'name',str(defender))} fouls out!\n"
                        break

                # offensive foul
                if random.random() < 0.03:
                    ball_handler.box.foul = getattr(ball_handler.box, "foul", 0) + 1
                    actions += f"{getattr(ball_handler,'name',str(ball_handler))} commits an offensive foul!\n"
                    self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                    # attempt substitution safely; ignore return
                    try:
                        self.perform_substitution(team, actions, players_on_court, team.original_starters)
                    except Exception:
                        pass
                    break

                # small steal chance while dribbling
                steal_weights = []
                for p in defenders_on_court:
                    gp = safe_gp(p)
                    spg = safe_attr(p, "spg", 0.0)
                    steal_weights.append(max(spg / gp + 0.01, 0.001))

                if defenders_on_court and sum(steal_weights) > 0 and random.random() < 0.03:
                    stealer = random.choices(defenders_on_court, weights=steal_weights, k=1)[0]
                    stealer.box.stl += 1
                    ball_handler.box.tov += 1
                    actions += f"{getattr(stealer,'name',str(stealer))} steals the ball from {getattr(ball_handler,'name',str(ball_handler))} while dribbling!\n"
                    result = "steal"
                    self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                    break

                actions += f"{getattr(ball_handler,'name',str(ball_handler))} dribbles ({action_time}s)\n"

            # --- SHOT ACTION ---
            elif action == "shot" or last_shot:
                # possible foul before shot
                for defender in defenders_on_court:
                    fta_rate = min(safe_attr(ball_handler, "FTA", 0.0), 400)
                    normalized_rate = min(fta_rate / 6.0, 2.0)
                    foul_tendency = 1 / (1 + math.exp(-2.0 * (normalized_rate - 0.8)))
                    bias_factor = 1 + normalized_rate ** 2 * 0.4
                    foul_factor = (0.015 + foul_tendency * 0.2) * bias_factor

                    if random.random() < foul_factor * 0.08:
                        defender.box.foul = getattr(defender.box, "foul", 0) + 1
                        actions += f"{getattr(defender,'name',str(defender))} commits a foul on {getattr(ball_handler,'name',str(ball_handler))}!\n"
                        free_foul = True
                        break

                shooter = ball_handler
                shooter_FG3A = safe_attr(shooter, "FG3A", 0.0)
                shooter_FGA = safe_attr(shooter, "FGA", 0.0)
                three_point_rate = shooter_FG3A / max(shooter_FGA, 1) if shooter_FGA else 0.3
                is_three = random.random() < three_point_rate
                points = 3 if is_three else 2

                shooter_FG3_PCT = safe_attr(shooter, "FG3_PCT", None)
                shooter_FG_PCT = safe_attr(shooter, "FG_PCT", None)
                fg_chance = shooter_FG3_PCT if (is_three and shooter_FG3_PCT) else (shooter_FG_PCT if shooter_FG_PCT else 0.45)
                fg_chance = fg_chance if fg_chance else 0.45

                # block weights
                block_weights = []
                for p in defenders_on_court:
                    gp = safe_gp(p)
                    bpg = safe_attr(p, "bpg", 0.0)
                    height = safe_attr(p, "height", 0)
                    pos = getattr(p, "position", "").lower()
                    w = (bpg / gp) * 3 + 0.01
                    if bpg > 1:
                        w *= 3
                    if pos and pos[0] == "c":
                        w *= 3
                    if height and height > 81:
                        w *= 3
                    block_weights.append(max(w, 0.001))

                if defenders_on_court and sum(block_weights) > 0 and random.random() < 0.05:
                    points = 0
                    blocker = random.choices(defenders_on_court, weights=block_weights, k=1)[0]
                    blocker.box.blk += 1
                    shooter.box.fga = getattr(shooter.box, "fga", 0) + 1
                    actions += f"{getattr(blocker,'name',str(blocker))} blocks the shot of {getattr(shooter,'name',str(shooter))}!\n"
                    result = "block"
                    self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                    break

                made = random.random() < (fg_chance / max(random.gauss(0.95, 0.15), 0.1))
                if made:
                    actions += f"{getattr(shooter,'name',str(shooter))} makes a {points}-point shot! ({action_time}s)\n"
                    result = "score"
                    shooter.box.fgm = getattr(shooter.box, "fgm", 0) + 1
                    shooter.box.fga = getattr(shooter.box, "fga", 0) + 1
                    shooter.box.points = getattr(shooter.box, "points", 0) + points
                    if is_three:
                        shooter.box.tpm = getattr(shooter.box, "tpm", 0) + 1
                        shooter.box.tpa = getattr(shooter.box, "tpa", 0) + 1
                        shooter.box.tpp = round(shooter.box.tpm / max(shooter.box.tpa, 1), 3)
                    if passes > 0 and last_passer and last_passer is not shooter:
                        last_passer.box.ast = getattr(last_passer.box, "ast", 0) + 1

                    if self.curr_possession == 'Team2':
                        self.team2_score += points
                    else:
                        self.team1_score += points

                    if free_foul:
                        actions += f"{getattr(shooter,'name',str(shooter))} is fouled and heads to the free throw line (1 attempt) AND-ONE!\n"
                        actions = self.free_throw(shooter, 1, actions)
                        self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                        break

                    for p in defenders_on_court:
                        p.box.plus_minus = getattr(p.box, "plus_minus", 0) - points
                    for p in players_on_court:
                        p.box.plus_minus = getattr(p.box, "plus_minus", 0) + points

                else:
                    actions += f"{getattr(shooter,'name',str(shooter))} misses a {points}-point shot ({action_time}s)\n"
                    shooter.box.fga = getattr(shooter.box, "fga", 0) + 1
                    if is_three:
                        shooter.box.tpa = getattr(shooter.box, "tpa", 0) + 1
                        shooter.box.tpp = round(getattr(shooter.box, "tpm", 0) / max(getattr(shooter.box, "tpa", 1), 1), 3)

                    if free_foul:
                        actions += f"{getattr(shooter,'name',str(shooter))} is fouled and heads to the free throw line:  {points} attempts! \n"
                        actions = self.free_throw(shooter, points, actions)
                        self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                        break

                    # Offensive rebound attempt
                    total_team_rpg = sum(max(safe_attr(p, "rpg", 0.0), 0.0) for p in players_on_court)
                    total_all_rpg = sum(max(safe_attr(p, "rpg", 0.0), 0.0) for p in (players_on_court + defenders_on_court))
                    if total_all_rpg <= 0:
                        total_all_rpg = 1.0

                    team_off_reb_rate = (total_team_rpg * random.gauss(0.4, 0.15) / max(1.0, total_all_rpg))
                    # clamp to [0,1]
                    team_off_reb_rate = max(0.0, min(team_off_reb_rate, 1.0))

                    if random.random() < team_off_reb_rate and players_on_court:
                        # Offensive rebound
                        reb_weights = []
                        for p in players_on_court:
                            rpg = safe_attr(p, "rpg", 0.0)
                            w = (rpg * 1.5) if rpg > 6.5 else max(rpg, 3)
                            reb_weights.append(w)
                        if sum(reb_weights) <= 0:
                            reb_weights = [1] * len(players_on_court)
                        rebounder = random.choices(players_on_court, weights=reb_weights, k=1)[0]
                        rebounder.box.oreb = getattr(rebounder.box, "oreb", 0) + 1
                        actions += f"{getattr(rebounder,'name',str(rebounder))} grabs the offensive rebound!\n"
                        shot_clock = 14
                        ball_handler = rebounder
                        continue

                    else:
                        # Defensive rebound
                        d_weights = []
                        for p in defenders_on_court:
                            rpg = safe_attr(p, "rpg", 0.0)
                            w = (rpg * 1.2) if rpg > 6.5 else max(rpg, 5)
                            d_weights.append(w)
                        if not defenders_on_court or sum(d_weights) <= 0:
                            d_weights = [1] * len(defenders_on_court) if defenders_on_court else [1]
                        d_rebounder = random.choices(defenders_on_court, weights=d_weights, k=1)[0]
                        d_rebounder.box.dreb = getattr(d_rebounder.box, "dreb", 0) + 1
                        actions += f"{getattr(d_rebounder,'name',str(d_rebounder))} grabs the defensive rebound!\n"
                        self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                        ball_handler = d_rebounder
                        break

                # finalize shot bookkeeping
                shooter.box.fgp = round(getattr(shooter.box, "fgm", 0) / max(getattr(shooter.box, "fga", 1), 1), 3)
                self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                break

            # --- TURNOVER ACTION ---
            elif action == "turnover":
                actions += f"{getattr(ball_handler,'name',str(ball_handler))} turns the ball over ({action_time}s)\n"
                ball_handler.box.tov = getattr(ball_handler.box, "tov", 0) + 1
                result = "turnover"
                self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                break

        # --- Update timers ---
        for p in players_on_court + defenders_on_court:
            p.total_seconds = getattr(p, "total_seconds", 0) + possession_time
            p.stint_seconds = getattr(p, "stint_seconds", 0) + possession_time

        for p in getattr(team, "bench", []) + getattr(defense, "bench", []):
            p.bench_stint_seconds = getattr(p, "bench_stint_seconds", 0) + possession_time

        # --- Perform substitutions ---
        MIN_POSSESSIONS_BETWEEN_SUBS = 8
        if (self.poss_counter1 - self.last_sub_poss1) >= MIN_POSSESSIONS_BETWEEN_SUBS:
            actions = self.perform_substitution(self.team1, actions, self.team1.starters, self.team1.original_starters)
            self.last_sub_poss1 = self.poss_counter1
        if (self.poss_counter2 - self.last_sub_poss2) >= MIN_POSSESSIONS_BETWEEN_SUBS:
            actions = self.perform_substitution(self.team2, actions, self.team2.starters, self.team2.original_starters)
            self.last_sub_poss2 = self.poss_counter2

        actions += (f"=== Current Score ===\nTeam1: {self.team1_score} Team2: {self.team2_score}\n")
        self.occurances.append(actions)

        return possession_time




def normalize_name(name):
    # Remove accents and trim spaces
    return ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    ).strip()


def build_cumulative_data(team, year, json_folder="seasons_json"):
    """
    Build cumulative stats and awards for a team over the previous 3 seasons.
    Handles rookies with no previous data by keeping positions/height/weight.
    """
    start = int(year.split("-")[0])
    end = int(year.split("-")[1])
    previous_seasons = [f"{start - i}-{str(end - i)[-2:]}" for i in range(1, 4)][::-1]

    cumulative_data = {}

    for player in team:
        cumulative_data[player] = {
            "awards": {key: 0 for key in [
                "MVP","All-NBA-First","All-NBA-Second","All-NBA-Third","NBA All-Star Game MVP",
                "NBA All-Star","DPOY","All-Defensive-First","All-Defensive-Second","ROY",
                "All-Rookie-First","All-Rookie-Second","Championships","FMVP"
            ]},
            "stats": {
                "total_games": 0, "PPG": 0.0, "APG": 0.0, "RBG": 0.0, "SPG": 0.0,
                "BPG": 0.0, "TPG": 0.0, "MPG": 0.0, "FGM": 0, "FGA": 0,
                "FG3M": 0, "FG3A": 0, "FTM": 0, "FTA": 0, "FG_PCT": 0.0,
                "FG3_PCT": 0.0, "FT_PCT": 0.0
            },
            "Position": None,
            "height": 0,
            "weight": 0
        }

        # Load season data
        player_seasons = {}
        player_found = False
        player_norm = normalize_name(player)

        for season in previous_seasons:
            json_path = os.path.join(json_folder, f"{season}.json")
            if not os.path.exists(json_path):
                player_seasons[season] = {}
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                season_data_raw = json.load(f)

            season_data = {normalize_name(k): v for k, v in season_data_raw.items()}

            if player_norm in season_data:
                player_seasons[season] = season_data[player_norm]
                player_found = True
            else:
                close = get_close_matches(player_norm, season_data.keys(), n=1, cutoff=0.85)
                if close:
                    match = close[0]
                    print(f"⚠️  Fuzzy match: '{player}' → '{match}' ({season})")
                    player_seasons[season] = season_data[match]
                    player_found = True
                else:
                    player_seasons[season] = {}

        # Rookie handling: if no data at all, still set position/height/weight
        if not player_found:
            json_path = os.path.join(json_folder, f"{year}.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    season_data_raw = json.load(f)
                season_data = {normalize_name(k): v for k, v in season_data_raw.items()}
                player_info = season_data.get(player_norm, {})
                cumulative_data[player]["Position"] = player_info.get("position", "Unknown")
                cumulative_data[player]["height"] = player_info.get("height", 0)
                cumulative_data[player]["weight"] = player_info.get("weight", 0)

                continue  # skip stats/awards

        # Sum awards and stats
        for season, season_data in player_seasons.items():
            awards = season_data.get("awards", {})
            stats = season_data.get("stats", {})

            for key in cumulative_data[player]["awards"]:
                cumulative_data[player]["awards"][key] += int(awards.get(key, 0) or awards.get(key, False))

            GP = stats.get("GP", 0)
            cumulative_data[player]["stats"]["total_games"] += GP
            cumulative_data[player]["stats"]["PPG"] += (stats.get("3PM",0)*3 + (stats.get("FGM",0)-stats.get("3PM",0))*2 + stats.get("FTM",0))
            cumulative_data[player]["stats"]["APG"] += stats.get("APG",0) * GP
            cumulative_data[player]["stats"]["RBG"] += stats.get("RBG",0) * GP
            cumulative_data[player]["stats"]["SPG"] += stats.get("SPG",0) * GP
            cumulative_data[player]["stats"]["BPG"] += stats.get("BPG",0) * GP
            cumulative_data[player]["stats"]["TPG"] += stats.get("TPG",0) * GP
            cumulative_data[player]["stats"]["MPG"] += stats.get("MPG",0) * GP
            cumulative_data[player]["stats"]["FGM"] += stats.get("FGM",0)
            cumulative_data[player]["stats"]["FGA"] += stats.get("FGA",0)
            cumulative_data[player]["stats"]["FG3M"] += stats.get("FG3M",0)
            cumulative_data[player]["stats"]["FG3A"] += stats.get("FG3A",0)
            cumulative_data[player]["stats"]["FTM"] += stats.get("FTM",0)
            cumulative_data[player]["stats"]["FTA"] += stats.get("FTA",0)

            cumulative_data[player]["Position"] = season_data.get("position","Unknown")
            cumulative_data[player]["height"] = season_data.get("height",0)
            cumulative_data[player]["weight"] = season_data.get("weight",0)

        # Compute per-game averages
        s = cumulative_data[player]["stats"]
        total_games = s["total_games"]
        if total_games > 0:
            s["PPG"] = round(s["PPG"] / total_games, 3)
            s["APG"] = round(s["APG"] / total_games, 3)
            s["RBG"] = round(s["RBG"] / total_games, 3)
            s["SPG"] = round(s["SPG"] / total_games, 3)
            s["BPG"] = round(s["BPG"] / total_games, 3)
            s["TPG"] = round(s["TPG"] / total_games, 3)
            s["MPG"] = round(s["MPG"] / total_games, 3)
            s["FG_PCT"] = round(s["FGM"] / s["FGA"], 3) if s["FGA"] > 0 else 0
            s["FG3_PCT"] = round(s["FG3M"] / s["FG3A"], 3) if s["FG3A"] > 0 else 0
            s["FT_PCT"] = round(s["FTM"] / s["FTA"], 3) if s["FTA"] > 0 else 0

    return cumulative_data


if __name__ == "__main__":

    year = "2016-17"

    # 2016-17 NBA Rosters

    nba_2016_17 = {
        "Boston Celtics": [
            "Isaiah Thomas", "Avery Bradley", "Al Horford", "Jae Crowder", "Marcus Smart",
            "Kelly Olynyk", "Amir Johnson", "Terry Rozier", "Jaylen Brown", "Gerald Green", "Jonas Jerebko"
        ],
        "Brooklyn Nets": [
            "Jeremy Lin", "Brook Lopez", "Rondae Hollis-Jefferson", "Sean Kilpatrick", "Trevor Booker",
            "Caris LeVert", "Justin Hamilton", "Isaiah Whitehead", "Joe Harris", "Spencer Dinwiddie"
        ],
        "New York Knicks": [
            "Derrick Rose", "Carmelo Anthony", "Kristaps Porzingis", "Courtney Lee", "Joakim Noah",
            "Willy Hernangomez", "Brandon Jennings", "Lance Thomas", "Kyle O'Quinn", "Justin Holiday"
        ],
        "Philadelphia 76ers": [
            "Jahlil Okafor", "Joel Embiid", "Dario Saric", "Robert Covington", "T.J. McConnell",
            "Sergio Rodriguez", "Gerald Henderson", "Ersan Ilyasova", "Nik Stauskas", "Richaun Holmes"
        ],
        "Toronto Raptors": [
            "Kyle Lowry", "DeMar DeRozan", "Serge Ibaka", "Jonas Valanciunas", "Norman Powell",
            "Cory Joseph", "P.J. Tucker", "Terrence Ross", "Patrick Patterson", "Lucas Nogueira"
        ],
        "Chicago Bulls": [
            "Rajon Rondo", "Jimmy Butler", "Dwyane Wade", "Robin Lopez", "Nikola Mirotic",
            "Bobby Portis", "Jerian Grant", "Michael Carter-Williams", "Denzel Valentine", "Cristiano Felicio"
        ],
        "Cleveland Cavaliers": [
            "Kyrie Irving", "LeBron James", "Kevin Love", "Tristan Thompson", "Iman Shumpert",
            "Richard Jefferson", "Channing Frye", "J.R. Smith", "Kyle Korver", "Deron Williams"
        ],
        "Detroit Pistons": [
            "Reggie Jackson", "Andre Drummond", "Tobias Harris", "Kentavious Caldwell-Pope", "Stanley Johnson",
            "Marcus Morris", "Jon Leuer", "Ish Smith", "Aron Baynes", "Boban Marjanovic"
        ],
        "Indiana Pacers": [
            "Paul George", "Jeff Teague", "Myles Turner", "Monta Ellis", "Thaddeus Young",
            "C.J. Miles", "Glenn Robinson III", "Al Jefferson", "Aaron Brooks", "Lavoy Allen"
        ],
        "Milwaukee Bucks": [
            "Giannis Antetokounmpo", "Malcolm Brogdon", "Jabari Parker", "Matthew Dellavedova", "Tony Snell",
            "John Henson", "Mirza Teletovic", "Greg Monroe", "Michael Beasley", "Thon Maker"
        ],
        "Atlanta Hawks": [
            "Dennis Schroder", "Paul Millsap", "Dwight Howard", "Kent Bazemore", "Thabo Sefolosha",
            "Kyle Korver", "Tim Hardaway Jr.", "Mike Muscala", "Kris Humphries", "Taurean Prince"
        ],
        "Charlotte Hornets": [
            "Kemba Walker", "Nicolas Batum", "Michael Kidd-Gilchrist", "Marvin Williams", "Cody Zeller",
            "Frank Kaminsky", "Jeremy Lamb", "Spencer Hawes", "Marco Belinelli", "Brian Roberts"
        ],
        "Miami Heat": [
            "Goran Dragic", "Hassan Whiteside", "Dion Waiters", "Josh Richardson", "James Johnson",
            "Tyler Johnson", "Wayne Ellington", "Luke Babbitt", "Rodney McGruder", "Willie Reed"
        ],
        "Orlando Magic": [
            "Elfrid Payton", "Nikola Vucevic", "Aaron Gordon", "Evan Fournier", "Terrence Ross",
            "Bismack Biyombo", "Jeff Green", "Mario Hezonja", "D.J. Augustin", "C.J. Watson"
        ],
        "Washington Wizards": [
            "John Wall", "Bradley Beal", "Otto Porter Jr.", "Markieff Morris", "Marcin Gortat",
            "Kelly Oubre Jr.", "Trey Burke", "Tomas Satoransky", "Jason Smith", "Bojan Bogdanovic"
        ],
        "Denver Nuggets": [
            "Jamal Murray", "Nikola Jokic", "Gary Harris", "Wilson Chandler", "Danilo Gallinari",
            "Emmanuel Mudiay", "Jusuf Nurkic", "Kenneth Faried", "Will Barton", "Mason Plumlee"
        ],
        "Minnesota Timberwolves": [
            "Ricky Rubio", "Andrew Wiggins", "Karl-Anthony Towns", "Zach LaVine", "Gorgui Dieng",
            "Shabazz Muhammad", "Brandon Rush", "Cole Aldrich", "Nemanja Bjelica", "Kris Dunn"
        ],
        "Oklahoma City Thunder": [
            "Russell Westbrook", "Victor Oladipo", "Steven Adams", "Enes Kanter", "Andre Roberson",
            "Taj Gibson", "Doug McDermott", "Jerami Grant", "Domantas Sabonis", "Norris Cole"
        ],
        "Portland Trail Blazers": [
            "Damian Lillard", "C.J. McCollum", "Al-Farouq Aminu", "Jusuf Nurkic", "Maurice Harkless",
            "Noah Vonleh", "Ed Davis", "Evan Turner", "Allen Crabbe", "Shabazz Napier"
        ],
        "Utah Jazz": [
            "Gordon Hayward", "Rudy Gobert", "George Hill", "Rodney Hood", "Derrick Favors",
            "Joe Johnson", "Alec Burks", "Shelvin Mack", "Joe Ingles", "Boris Diaw"
        ],
        "Golden State Warriors": [
            "Stephen Curry", "Klay Thompson", "Kevin Durant", "Draymond Green", "Zaza Pachulia",
            "Andre Iguodala", "Shaun Livingston", "David West", "JaVale McGee", "Ian Clark"
        ],
        "Los Angeles Clippers": [
            "Chris Paul", "Blake Griffin", "DeAndre Jordan", "J.J. Redick", "Austin Rivers",
            "Jamal Crawford", "Luc Mbah a Moute", "Raymond Felton", "Wesley Johnson", "Marreese Speights"
        ],
        "Los Angeles Lakers": [
            "D'Angelo Russell", "Jordan Clarkson", "Julius Randle", "Larry Nance Jr.", "Timofey Mozgov",
            "Luol Deng", "Brandon Ingram", "Nick Young", "Tarik Black", "Ivica Zubac"
        ],
        "Phoenix Suns": [
            "Eric Bledsoe", "Devin Booker", "T.J. Warren", "Marquese Chriss", "Tyson Chandler",
            "Jared Dudley", "Alan Williams", "Brandon Knight", "Leandro Barbosa", "Alex Len"
        ],
        "Sacramento Kings": [
            "DeMarcus Cousins", "Rudy Gay", "Darren Collison", "Ben McLemore", "Willie Cauley-Stein",
            "Ty Lawson", "Garrett Temple", "Skal Labissiere", "Anthony Tolliver", "Buddy Hield"
        ],
        "Dallas Mavericks": [
            "Deron Williams", "Wesley Matthews", "Harrison Barnes", "Dirk Nowitzki", "Andrew Bogut",
            "Seth Curry", "Dwight Powell", "Nerlens Noel", "Yogi Ferrell", "Devin Harris"
        ],
        "Houston Rockets": [
            "James Harden", "Eric Gordon", "Clint Capela", "Ryan Anderson", "Trevor Ariza",
            "Lou Williams", "PJ Tucker", "Nene", "Patrick Beverley", "Sam Dekker"
        ],
        "Memphis Grizzlies": [
            "Mike Conley", "Marc Gasol", "Zach Randolph", "Tony Allen", "Chandler Parsons",
            "JaMychal Green", "Vince Carter", "James Ennis", "Wayne Selden", "Andrew Harrison"
        ],
        "New Orleans Pelicans": [
            "Jrue Holiday", "Anthony Davis", "DeMarcus Cousins", "ETwaun Moore", "Solomon Hill",
            "Terrence Jones", "Dante Cunningham", "Tim Frazier", "Buddy Hield", "Langston Galloway"
        ],
        "San Antonio Spurs": [
            "Tony Parker", "Kawhi Leonard", "LaMarcus Aldridge", "Pau Gasol", "Danny Green",
            "Manu Ginobili", "Patty Mills", "Jonathon Simmons", "Dewayne Dedmon", "David Lee"
        ]
    }


    season_data_storage = {}

    # --------------------------
    # Teams
    # --------------------------
    west_teams = [
        "Golden State Warriors", "Los Angeles Clippers", "Los Angeles Lakers", "Sacramento Kings", "Phoenix Suns",
        "San Antonio Spurs", "Houston Rockets", "Dallas Mavericks", "Memphis Grizzlies", "New Orleans Pelicans",
        "Oklahoma City Thunder", "Portland Trail Blazers", "Denver Nuggets", "Utah Jazz", "Minnesota Timberwolves"
    ]

    east_teams = [
        "Cleveland Cavaliers", "Boston Celtics", "Toronto Raptors", "Washington Wizards", "Atlanta Hawks",
        "Chicago Bulls", "Milwaukee Bucks", "Indiana Pacers", "Detroit Pistons", "Charlotte Hornets",
        "Miami Heat", "New York Knicks", "Philadelphia 76ers", "Brooklyn Nets", "Orlando Magic"
    ]

    # Divisions (5 teams each)
    west_divisions = [
        ["Golden State Warriors", "Los Angeles Clippers", "Los Angeles Lakers", "Sacramento Kings", "Phoenix Suns"],
        ["San Antonio Spurs", "Houston Rockets", "Dallas Mavericks", "Memphis Grizzlies", "New Orleans Pelicans"],
        ["Oklahoma City Thunder", "Portland Trail Blazers", "Denver Nuggets", "Utah Jazz", "Minnesota Timberwolves"]
    ]

    east_divisions = [
        ["Cleveland Cavaliers", "Boston Celtics", "Toronto Raptors", "Washington Wizards", "Atlanta Hawks"],
        ["Chicago Bulls", "Milwaukee Bucks", "Indiana Pacers", "Detroit Pistons", "Charlotte Hornets"],
        ["Miami Heat", "New York Knicks", "Philadelphia 76ers", "Brooklyn Nets", "Orlando Magic"]
    ]

    all_teams = west_teams + east_teams
    #print(all_teams)

    #try for Golden State:

    all_teams = west_teams + east_teams
    schedule = {t: {} for t in all_teams}

    def add_matchup(a, b, games):
        if a == b:
            return
        schedule[a][b] = games
        schedule[b][a] = games

    # Step 1: Inter-conference games (2 each)
    for w in west_teams:
        for e in east_teams:
            add_matchup(w, e, 2)

    def arrange_team(conference, divisions):
        for team in conference:
            # Find the division this team belongs to
            team_division = None
            for division in divisions:
                if team in division:
                    team_division = division
                    break
            
            if team_division is None:
                continue  # just in case

            # Index of the team inside its division
            team_idx = team_division.index(team)

            # --- Intra-division matchups (4 games each) ---
            for team_a in team_division:
                if team_a != team:
                    add_matchup(team, team_a, 4)

            # --- Cross-division matchups (mix of 3/4 games) ---
            for division in divisions:
                if division is not team_division:
                    for i in range(5):  # should be 5
                        other_team = division[i]
                        # Avoid same-team
                        if other_team == team:
                            continue
                        # Example logic: alternate 4 and 3 games by index pattern
                        if (team_idx + i) % 5 < 3:
                            add_matchup(other_team, team, 4)
                        else:
                            add_matchup(other_team, team, 3)
    

    def print_team_info(team, name):
        print(f"\n{name} lineup preview:")
        if hasattr(team, "players"):
            for p in team.players:
                print(f"  {getattr(p, 'name', str(p))}")
        elif hasattr(team, "roster"):
            for p in team.roster:
                print(f"  {getattr(p, 'name', str(p))}")
        else:
            print("  ⚠️ No player attribute found — check Team class definition")



    #Schedule now correctly contains every matchup each team has with opposing teams correctly!
    arrange_team(west_teams,west_divisions)
    arrange_team(east_teams,east_divisions)


team_1 = "Golden State Warriors"
team_2 = "Brooklyn Nets"

team_1_players = nba_2016_17.get(team_1, [])
team_2_players = nba_2016_17.get(team_2, [])
team_1_data = build_cumulative_data(team_1_players, year)
team_2_data = build_cumulative_data(team_2_players, year)

team1 = determine_seconds_and_make_team(team_1_data)
team2 = determine_seconds_and_make_team(team_2_data)

#print(67)

for player in team1.starters:
    print(f"{player.name} merit = {player.merit} position = {player.position}")

for player in team1.bench:
    print(f"{player.name} merit = {player.merit} position = {player.position}")

for player in team2.starters:
    print(f"{player.name} merit = {player.merit} position = {player.position}")

for player in team2.bench:
    print(f"{player.name} merit = {player.merit} position = {player.position}")

game = Game(team1, team2, 48)
game.tip_off()

try:
    game.simulate_full_game()
except ValueError as e:
    print(f"⚠️ Game failed: {team_1} vs {team_2} — {e}")


#RUNS AN ENTIRE SEASON! 

'''
# Initialize tracking dictionaries
team_records = {team: {"W": 0, "L": 0} for team in nba_2016_17.keys()}
player_stats_accum = {}  # key: player_name, value: cumulative stats dict
player_games_played = {}  # key: player_name, value: number of games played
played_games = set()

for team_a, opponents in schedule.items():
    for team_b, games in opponents.items():
        # Normalize team pair (alphabetical order)
        matchup_key = tuple(sorted([team_a, team_b]))
        if matchup_key in played_games:
            continue  # already simulated this matchup

        played_games.add(matchup_key)

        for game_num in range(1, games + 1):
            print(f"Simulating {team_a} vs {team_b}, Game {game_num}")

            team_1_players = nba_2016_17.get(team_a, [])
            team_2_players = nba_2016_17.get(team_b, [])
            team_1_data = build_cumulative_data(team_1_players, year)
            team_2_data = build_cumulative_data(team_2_players, year)

            team1 = determine_seconds_and_make_team(team_1_data)
            team2 = determine_seconds_and_make_team(team_2_data)

            game = Game(team1, team2, 48)
            game.tip_off()

            try:
                game.simulate_full_game()
            except ValueError as e:
                print(f"⚠️ Game failed: {team_a} vs {team_b} — {e}")
                continue

            # --- Update team records ---
            if game.team1_score > game.team2_score:
                team_records[team_a]["W"] += 1
                team_records[team_b]["L"] += 1
            elif game.team2_score > game.team1_score:
                team_records[team_b]["W"] += 1
                team_records[team_a]["L"] += 1
            else:
                # optional: handle tie if your simulation allows
                pass

            # --- Update player stats ---
            for team in [team1, team2]:
                for p in team.starters + team.bench:  # include starters and bench
                    name = getattr(p, "name", str(p))
                    if name not in player_stats_accum:
                        # initialize cumulative stats
                        player_stats_accum[name] = {
                            "points": 0, "fgm": 0, "fga": 0, "ast": 0,
                            "reb": 0, "oreb": 0, "dreb": 0, "stl": 0,
                            "blk": 0, "tov": 0, "fouls": 0, "minutes": 0
                        }
                        player_games_played[name] = 0

                    stats = player_stats_accum[name]
                    stats["points"] += getattr(p.box, "points", 0)
                    stats["fgm"] += getattr(p.box, "fgm", 0)
                    stats["fga"] += getattr(p.box, "fga", 0)
                    stats["ast"] += getattr(p.box, "ast", 0)
                    stats["reb"] += getattr(p.box, "oreb", 0) + getattr(p.box, "dreb", 0)
                    stats["oreb"] += getattr(p.box, "oreb", 0)
                    stats["dreb"] += getattr(p.box, "dreb", 0)
                    stats["stl"] += getattr(p.box, "stl", 0)
                    stats["blk"] += getattr(p.box, "blk", 0)
                    stats["tov"] += getattr(p.box, "tov", 0)
                    stats["fouls"] += getattr(p.box, "foul", 0)
                    stats["minutes"] += getattr(p, "total_seconds", 0)

                    player_games_played[name] += 1

# --- Compute per-game averages ---
player_stats_avg = {}

for name, stats in player_stats_accum.items():
    games = max(player_games_played.get(name, 1), 1)
    avg_stats = {}

    for k, v in stats.items():
        if k == "minutes":
            # Compute average seconds per game
            avg_secs = v / games
            minutes = int(avg_secs // 60)
            seconds = int(avg_secs % 60)
            avg_stats[k] = f"{minutes}:{seconds:02d}"
        else:
            avg_stats[k] = round(v / games, 1)
    
    player_stats_avg[name] = avg_stats


nba_dataset = nba_2016_17.copy()  # or deepcopy if needed

for team, players in nba_dataset.items():
    for i, player in enumerate(players):
        # Replace player name with a dict containing name + stats
        nba_dataset[team][i] = {
            "name": player,
            "stats": player_stats_avg.get(player, {})  # safely get stats
        }


# Make sure directory exists
os.makedirs("simulation_results", exist_ok=True)

# Save team records
with open("simulation_results/team_records.json", "w", encoding="utf-8") as f:
    json.dump(team_records, f, indent=4)

# Save cumulative player stats
with open("simulation_results/player_stats_accum.json", "w", encoding="utf-8") as f:
    json.dump(player_stats_accum, f, indent=4)

# Save number of games played per player
with open("simulation_results/player_games_played.json", "w", encoding="utf-8") as f:
    json.dump(player_games_played, f, indent=4)

with open("simulation_results/player_stats_avg.json", "w", encoding="utf-8") as f:
    json.dump(player_stats_avg, f, indent=4)

with open("simulation_results/final_dataset.json", "w", encoding="utf-8") as f:
    json.dump(nba_dataset, f, indent=4)
# Now you have:
# team_records: W/L for each team
# player_stats_avg: average stats per game for each player

'''


                    

                    




            








    


