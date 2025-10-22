#import json
import pandas as pd
import random
import math
import os
import json
import numpy as np

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
        self.original_starters = None
        self.bench = [p for p in self.players.values() if p.role == 'Bench']
        self.original_bench = None
        self.wins = 0
        self.losses = 0
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

    

def determine_seconds_and_make_team(team_data, top_bench_boost=3):
    import math

    # --- Initialize team ---
    team = Team(team_data)

    # --- Calculate player merit ---
    for player_name, player_info in team_data.items():
        awards = player_info.get('awards', {})
        stats = player_info.get('stats', {})

        award_merit_sum = ((stats.get('total_games', 0) ** (1/3)) +
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

        points_contributed = stats.get('PPG', 0) * stats.get('total_games', 0)
        assists_contributed = stats.get('APG', 0) * stats.get('total_games', 0)
        if stats.get('FG3_PCT', 0) > 0.38 and stats.get('FG3M', 0) > 100:
            points_contributed *= 1.5
        offensive_merit = math.sqrt(points_contributed * assists_contributed)

        steals_generated = stats.get('SPG', 0) * stats.get('total_games', 0)
        blocks_generated = stats.get('BPG', 0) * stats.get('total_games', 0)
        turnovers_generated = stats.get('TPG', 0) * stats.get('total_games', 0)
        defensive_merit = math.sqrt(steals_generated * blocks_generated) - turnovers_generated / 10

        merit = int(award_merit_sum * 0.15 + offensive_merit * 0.55 + defensive_merit * 0.45)
        merit = max(merit, 60)

        if player_name in team.players:
            team.players[player_name].merit = merit

    # --- Choose starters ---
    choose_best_formation(team)

    # --- Assign roles ---
    for player in team.players.values():
        player.role = "Starter" if player in team.starters else "Bench"

    # --- Parameters ---
    min_starter_seconds = 25 * 60
    max_starter_seconds = 42 * 60
    min_bench_seconds = 0 * 60
    max_bench_seconds = 32 * 60
    target_starters_total = 160 * 60  # 5 starters
    target_bench_total = 80 * 60      # bench total

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
        MAX_STINT = 60 * random.randint(12, 24)
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

        
        players_on_court = team.starters
        defenders_on_court = defense.starters
        players_bench = team.bench
        defenders_bench = defense.bench

        #adjust tendencies for players to have the ball
        total_team_apg = sum(p.apg for p in players_on_court)
        pass_weights = []
        for p in players_on_court:
            # Fraction of team's assists player typically accounts for, plus small baseline
            weight = (p.apg ** 3 / max(total_team_apg, 1)) * 0.7 + 0.3
            pass_weights.append(weight)

        # Pick initial ball handler based on weighted shot probability
        ball_handler = random.choices(players_on_court, weights=pass_weights, k=1)[0]

        last_passer = None
        passes = 0
        actions = f"{ball_handler.name} starts with the ball\n"

        #shot_clock = 24
        possession_time = 0
        last_passer = None
        points = 0

        rem = int(remaining_secs)

        while shot_clock > 0 and rem > 0:
            # Action time
            action_time = random.choices([1,2,3,4,5,6,7,8,9], weights=[0.025,0.1,0.15,0.225,0.225,0.15,0.075,0.025,0.025], k=1)[0]
            action_time = min(action_time, shot_clock, rem)
            last_shot = shot_clock <= action_time or rem <= action_time

            shot_clock -= action_time
            rem -= action_time
            possession_time += action_time

            # Update defenders
            defenders_on_court = defense.starters


            # --- Calculate probabilities ---
            turnover_prob = min(ball_handler.tpg / max(ball_handler.gp,1) * 0.06 + 0.012, 0.03)
            shooting_volume = min((ball_handler.FGM / max(ball_handler.gp,1))/20,1)
            shooting_efficiency = ball_handler.FGM / max(ball_handler.FGA,1) if ball_handler.FGA else random.uniform(0.35,0.55)

            # Low-volume boost for efficient shooters
            volume_f = min(2 * shooting_volume, 1)
            low_volume_boost = (1 - volume_f) * shooting_efficiency * 0.1

            score_weight = shooting_efficiency * volume_f + low_volume_boost
            assist_weight = min(ball_handler.apg / max(ball_handler.gp,1) * 70,0.8)

            # Normalize weights to probabilities
            total_offense = score_weight * 1.5 + assist_weight
            shot_prob = (score_weight / total_offense) * (1 - turnover_prob) * 0.6
            shot_prob = min(shot_prob,0.6)

            # Factor goes from 1 (lots of time) → 0 (no time left)
            time_factor = max(min(shot_clock, rem) / 24.0, 0)   # normalized 0–1

            # Dribbling also shrinks, but has a small baseline early in clock
            dribble_prob = max(min((0.035 * shot_clock), 0.22), 0.03) * time_factor

            # Passing shrinks with time
            #pass_prob = (assist_weight / total_offense) * (1 - turnover_prob) * 0.8 * time_factor
            apg_factor = min(ball_handler.apg / 10, 0.3) 
            pass_prob = (1 - shot_prob - turnover_prob - dribble_prob) * time_factor * (1 + apg_factor)

            # If no time left, force shot/turnover only
            if min(shot_clock, rem) <= 0:
                pass_prob = 0
                dribble_prob = 0


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
        
            free_foul = False

            # --- PASS ACTION ---
            if action == "pass":
                steal_weights = [math.pow(p.spg,3) / max(p.gp, 1) + 0.02 for p in defenders_on_court]
                if random.random() < 0.04:
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
                #defensive foul prob
                for defender in defenders_on_court:

                    #player bait odds are better if they attempt more free throws
                    foul_weight = ball_handler.FTA/ max(ball_handler.gp,1)
                    foul_tendency = 1 / (1 + math.exp(-0.6 * (foul_weight - 5)))
                    foul_factor = foul_tendency * 0.2

                    if random.random() < foul_factor * 0.08:  # 2% chance per defender per action
                        defender.box.foul += 1
    
                        actions += (f"{defender.name} commits a foul on {ball_handler.name}!\n")
                        
                        # Check for foul-out
                        if defender.box.foul == 6:
                            actions = self.perform_substitution(defense, actions, defense.starters,defense.original_starters)
                            actions += (f"{defender.name} fouls out!")
                        break


                #offensive foul prob
                offensive_foul_prob = 0.03
                if random.random() < offensive_foul_prob:
                    ball_handler.box.foul += 1
                    actions += f"{ball_handler.name} commits an offensive foul!\n"
                    self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                    if ball_handler.box.foul >= 6:
                        actions += (f"{ball_handler.name} fouls out!")
                        self.perform_substitution(team,actions,players_on_court,team.original_starters)
                    break

                steal_weights = [
                    ((p.spg / max(p.gp, 1)) * 3 + 0.01) * (10 if p.spg > 0.6 else 1)
                    for p in defenders_on_court
                ]
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
                for defender in defenders_on_court:

                    fta_rate = min(ball_handler.FTA,400)
                    normalized_rate = min(fta_rate / 6, 2)  # scale typical range (e.g. 0–12 FTA/game)
                    foul_tendency = 1 / (1 + math.exp(-2.0 * (normalized_rate - 0.8)))  # steeper sigmoid
                    bias_factor = 1 + normalized_rate ** 2 * 0.4  # amplify high-FTA players
                    foul_factor = (0.015 + foul_tendency * 0.2) * bias_factor

                    if random.random() < foul_factor * 0.1:
                        defender.box.foul += 1
                        actions += (f"{defender.name} commits a foul on {ball_handler.name}!")
                        if defender.box.foul == 6:
                            actions = self.perform_substitution(defense, actions, defense.starters,defense.original_starters)
                            actions += (f"{defender.name} fouls out!")
                        free_foul = True
                        break

                shooter = ball_handler
                three_point_rate = shooter.FG3A / max(shooter.FGA,1) if shooter.FGA else 0.1
                is_three = random.random() < three_point_rate
                points = 3 if is_three else 2
                fg_chance = shooter.FG3_PCT if is_three and shooter.FG3_PCT else shooter.FG_PCT
                fg_chance = fg_chance if fg_chance else 0.45

                block_weights = [
                    ((p.bpg / max(p.gp, 1)) * 3 + 0.01)                     # base BPG factor
                    * (3 if p.bpg > 1 else 1)                               # high-block bonus
                    * (3 if p.position[0].lower() == "c" else 1)            # center bonus
                    * (3 if getattr(p, "height", 0) > 81 else 1)            # height bonus (in inches)
                    for p in defenders_on_court
                ]



                if random.random() < 0.04:
                    points = 0
                    blocker = random.choices(defenders_on_court, weights=block_weights, k=1)[0]
                    blocker.box.blk += 1
                    shooter.box.fga += 1
                    actions += f"{blocker.name} blocks the shot of {shooter.name}!\n"
                    result = "block"
                    self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                    break

                made = random.random() < fg_chance / random.gauss(0.95,0.15)
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
                        # --- Update scores
                    if self.curr_possession == 'Team2':
                        self.team2_score += points
                    else:
                        self.team1_score += points

                    if free_foul is True:
                        #make 1 free throw
                        actions += f"{shooter.name} is fouled and heads to the free throw line(1 attempt) AND ONE!\n"
                        actions = self.free_throw(shooter,1,actions)
                        self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                        break

                     # and plus/minus ---
                    
                    for p in defenders_on_court:
                        p.box.plus_minus -= points
                    for p in players_on_court:
                        p.box.plus_minus += points
                        
                else:
                    actions += f"{shooter.name} misses a {points}-point shot ({action_time}s)\n"
                    shooter.box.fga += 1
                    if is_three:
                        shooter.box.tpa += 1
                        shooter.box.tpp = round(shooter.box.tpm / shooter.box.tpa,3)
                    if free_foul is True:
                        actions += f"{shooter.name} is fouled and heads to the free throw line {points} attempts\n"
                        actions = self.free_throw(shooter,points,actions)
                        self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                        break

                    #offensive rebound!
                    # --- Offensive rebound probability ---
                    #print(players_on_court)

                    total_team_rpg = sum(p.rpg for p in players_on_court)
                    total_def_rpg = sum(p.rpg for p in players_on_court + defenders_on_court)

                    team_off_reb_rate = (total_team_rpg * random.gauss(0.4, 0.15) 
                                        / max(1, total_def_rpg))

                    
                    if random.random() < team_off_reb_rate:
                        # Offensive rebound
                        rebounder = random.choices(players_on_court, weights=
                                                   [p.rpg ** 1.2 if p.rpg > 5.5 else p.rpg * 2.1 for p in players_on_court], k=1)[0]
                        rebounder.box.oreb += 1
                        actions += f"{rebounder.name} grabs the offensive rebound!\n"
                        # reset shot clock for continuation
                        shot_clock = 14
                        ball_handler = rebounder
                        continue

                    else:
                        # Defensive rebound
                        d_rebounder = random.choices(defenders_on_court, weights=
                                                   [p.rpg ** 1.5 if p.rpg > 5.5 else p.rpg * 2.1 for p in defenders_on_court], k=1)[0]
                        d_rebounder.box.dreb += 1
                        actions += f"{d_rebounder.name} grabs the defensive rebound!\n"
                        # possession switches
                        self.curr_possession = 'Team2' if self.curr_possession == 'Team1' else 'Team1'
                        ball_handler = d_rebounder
                        break  # possession ends

                        
                        
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



        # --- Update timers ---
        for p in players_on_court + defenders_on_court:
            p.total_seconds = getattr(p, "total_seconds", 0) + possession_time
            p.stint_seconds = getattr(p, "stint_seconds", 0) + possession_time

        for p in players_bench + defenders_bench:
            p.bench_stint_seconds = getattr(p, "bench_stint_seconds", 0) + possession_time

        # --- Perform substitutions --_
        MIN_POSSESSIONS_BETWEEN_SUBS = 6
        if (self.poss_counter1 - self.last_sub_poss1) >= MIN_POSSESSIONS_BETWEEN_SUBS:
            actions = self.perform_substitution(self.team1,actions,self.team1.starters,self.team1.original_starters)
            self.last_sub_poss1 = self.poss_counter1
        if (self.poss_counter2 - self.last_sub_poss2) >= MIN_POSSESSIONS_BETWEEN_SUBS:
            actions = self.perform_substitution(self.team2,actions,self.team2.starters,self.team2.original_starters)
            self.last_sub_poss2 = self.poss_counter2

        return possession_time




def build_cumulative_data(team, year, json_folder="seasons_json"):
    """
    Build cumulative stats and awards for a team over the previous 3 seasons.
    
    Args:
        team (list): list of player names
        year (str): current season in "YYYY-YY" format
        json_folder (str): folder containing season JSON files
    
    Returns:
        dict: cumulative stats and awards for each player
    """
    start = int(year.split("-")[0])
    end = int(year.split("-")[1])
    previous_seasons = [f"{start - i}-{str(end - i)[-2:]}" for i in range(1, 4)]
    
    cumulative_data = {}

    for player in team:
        # Initialize player cumulative data
        cumulative_data[player] = {
            "awards": {
                "MVP": 0,
                "All-NBA-First": 0,
                "All-NBA-Second": 0,
                "All-NBA-Third": 0,
                "NBA All-Star Game MVP": 0,
                "NBA All-Star": 0,
                "DPOY": 0,
                "All-Defensive-First": 0,
                "All-Defensive-Second": 0,
                "ROY": 0,
                "All-Rookie-First": 0,
                "All-Rookie-Second": 0,
                "Championships": 0,
                "FMVP": 0
            },
            "stats": {
                "total_games": 0,
                "PPG": 0.0,
                "APG": 0.0,
                "RBG": 0.0,
                "SPG": 0.0,
                "BPG": 0.0,
                "TPG": 0.0,
                "MPG": 0.0,
                "FGM": 0,
                "FGA": 0,
                "FG3M": 0,
                "FG3A": 0,
                "FTM": 0,
                "FTA": 0,
                "FG_PCT": 0.0,
                "FG3_PCT": 0.0,
                "FT_PCT": 0.0
            }
        }

        # Load season data
        player_seasons = {}
        for season in previous_seasons:
            json_path = os.path.join(json_folder, f"{season}.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    season_data = json.load(f)
                player_seasons[season] = season_data.get(player, {})
            else:
                player_seasons[season] = {}

        # Sum awards and stats
        for season, season_data in player_seasons.items():
            awards = season_data.get("awards", {})
            stats = season_data.get("stats", {})
            # Update awards
            for key in cumulative_data[player]["awards"]:
                cumulative_data[player]["awards"][key] += int(awards.get(key, 0) or awards.get(key, False))

            # Update stats
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
            #Update height/weight attributes
            cumulative_data[player]["Position"] = season_data.get("position","Unknown")
            cumulative_data[player]["height"] = season_data.get("height",0)
            cumulative_data[player]["weight"] = season_data.get("weight",0)

        # Compute percentages and per-game averages
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
        else:
            # No data, set everything to 0
            for key in s:
                s[key] = 0

        #last season!
    return cumulative_data

if __name__ == "__main__":

    year = "2016-17"

    # 2016-17 NBA Rosters

    nba_2016_17 = {
        "Boston Celtics": [
            "Isaiah Thomas", "Avery Bradley", "Al Horford", "Jae Crowder", "Marcus Smart",
            "Kelly Olynyk", "Amir Johnson", "Terry Rozier", "Jaylen Brown", "Gerald Green"
        ],
        "Brooklyn Nets": [
            "Jeremy Lin", "Brook Lopez", "Rondae Hollis-Jefferson", "Sean Kilpatrick", "Trevor Booker",
            "Caris LeVert", "Justin Hamilton", "Isaiah Whitehead"
        ],
        "New York Knicks": [
            "Derrick Rose", "Carmelo Anthony", "Kristaps Porzingis", "Courtney Lee", "Joakim Noah",
            "Willy Hernangomez", "Brandon Jennings", "Lance Thomas"
        ],
        "Philadelphia 76ers": [
            "Jahlil Okafor", "Joel Embiid", "Dario Saric", "Robert Covington", "T.J. McConnell",
            "Sergio Rodriguez", "Gerald Henderson"
        ],
        "Toronto Raptors": [
            "Kyle Lowry", "DeMar DeRozan", "Serge Ibaka", "Jonas Valanciunas", "Norman Powell",
            "Cory Joseph", "P.J. Tucker", "Terrence Ross"
        ],
        "Chicago Bulls": [
            "Rajon Rondo", "Jimmy Butler", "Dwyane Wade", "Robin Lopez", "Nikola Mirotic",
            "Bobby Portis", "Jerian Grant", "Michael Carter-Williams"
        ],
        "Cleveland Cavaliers": [
            "Kyrie Irving", "LeBron James", "Kevin Love", "Tristan Thompson", "J.R. Smith",
            "Iman Shumpert", "Richard Jefferson", "Channing Frye"
        ],
        "Detroit Pistons": [
            "Reggie Jackson", "Andre Drummond", "Tobias Harris", "Kentavious Caldwell-Pope", "Stanley Johnson",
            "Marcus Morris", "Jon Leuer", "Ish Smith"
        ],
        "Indiana Pacers": [
            "Paul George", "Jeff Teague", "Myles Turner", "Monta Ellis", "Thaddeus Young",
            "C.J. Miles", "Glenn Robinson III", "Al Jefferson"
        ],
        "Milwaukee Bucks": [
            "Giannis Antetokounmpo", "Malcolm Brogdon", "Jabari Parker", "Matthew Dellavedova", "Tony Snell",
            "John Henson", "Mirza Teletovic", "Greg Monroe"
        ],
        "Atlanta Hawks": [
            "Dennis Schroder", "Paul Millsap", "Dwight Howard", "Kent Bazemore", "Thabo Sefolosha",
            "Kyle Korver", "Tim Hardaway Jr.", "Mike Muscala"
        ],
        "Charlotte Hornets": [
            "Kemba Walker", "Nicolas Batum", "Michael Kidd-Gilchrist", "Marvin Williams", "Cody Zeller",
            "Frank Kaminsky", "Jeremy Lamb", "Spencer Hawes"
        ],
        "Miami Heat": [
            "Goran Dragic", "Hassan Whiteside", "Dion Waiters", "Josh Richardson", "James Johnson",
            "Tyler Johnson", "Wayne Ellington", "Luke Babbitt"
        ],
        "Orlando Magic": [
            "Elfrid Payton", "Nikola Vucevic", "Aaron Gordon", "Evan Fournier", "Terrence Ross",
            "Bismack Biyombo", "Jeff Green", "Mario Hezonja"
        ],
        "Washington Wizards": [
            "John Wall", "Bradley Beal", "Otto Porter Jr.", "Markieff Morris", "Marcin Gortat",
            "Kelly Oubre Jr.", "Trey Burke", "Tomas Satoransky"
        ],
        "Denver Nuggets": [
            "Jamal Murray", "Nikola Jokic", "Gary Harris", "Wilson Chandler", "Danilo Gallinari",
            "Emmanuel Mudiay", "Jusuf Nurkic", "Kenneth Faried"
        ],
        "Minnesota Timberwolves": [
            "Ricky Rubio", "Andrew Wiggins", "Karl-Anthony Towns", "Zach LaVine", "Gorgui Dieng",
            "Shabazz Muhammad", "Brandon Rush", "Cole Aldrich"
        ],
        "Oklahoma City Thunder": [
            "Russell Westbrook", "Victor Oladipo", "Steven Adams", "Enes Kanter", "Andre Roberson",
            "Taj Gibson", "Doug McDermott", "Jerami Grant"
        ],
        "Portland Trail Blazers": [
            "Damian Lillard", "C.J. McCollum", "Al-Farouq Aminu", "Jusuf Nurkic", "Maurice Harkless",
            "Noah Vonleh", "Ed Davis", "Evan Turner"
        ],
        "Utah Jazz": [
            "Gordon Hayward", "Rudy Gobert", "George Hill", "Rodney Hood", "Derrick Favors",
            "Joe Johnson", "Alec Burks", "Shelvin Mack"
        ],
        "Golden State Warriors": [
            "Stephen Curry", "Klay Thompson", "Kevin Durant", "Draymond Green", "Zaza Pachulia",
            "Andre Iguodala", "Shaun Livingston", "David West"
        ],
        "Los Angeles Clippers": [
            "Chris Paul", "Blake Griffin", "DeAndre Jordan", "J.J. Redick", "Austin Rivers",
            "Jamal Crawford", "Luc Mbah a Moute", "Raymond Felton"
        ],
        "Los Angeles Lakers": [
            "D'Angelo Russell", "Jordan Clarkson", "Julius Randle", "Larry Nance Jr.", "Timofey Mozgov",
            "Luol Deng", "Brandon Ingram", "Nick Young"
        ],
        "Phoenix Suns": [
            "Eric Bledsoe", "Devin Booker", "T.J. Warren", "Marquese Chriss", "Tyson Chandler",
            "Jared Dudley", "Alan Williams", "Brandon Knight"
        ],
        "Sacramento Kings": [
            "DeMarcus Cousins", "Rudy Gay", "Darren Collison", "Ben McLemore", "Willie Cauley-Stein",
            "Ty Lawson", "Garrett Temple", "Skal Labissiere"
        ],
        "Dallas Mavericks": [
            "Deron Williams", "Wesley Matthews", "Harrison Barnes", "Dirk Nowitzki", "Andrew Bogut",
            "Seth Curry", "Dwight Powell", "Nerlens Noel"
        ],
        "Houston Rockets": [
            "James Harden", "Eric Gordon", "Clint Capela", "Ryan Anderson", "Trevor Ariza",
            "Lou Williams", "PJ Tucker", "Nene"
        ],
        "Memphis Grizzlies": [
            "Mike Conley", "Marc Gasol", "Zach Randolph", "Tony Allen", "Chandler Parsons",
            "JaMychal Green", "Vince Carter", "James Ennis"
        ],
        "New Orleans Pelicans": [
            "Jrue Holiday", "Anthony Davis", "DeMarcus Cousins", "ETwaun Moore", "Solomon Hill",
            "Terrence Jones", "Dante Cunningham", "Tim Frazier"
        ],
        "San Antonio Spurs": [
            "Tony Parker", "Kawhi Leonard", "LaMarcus Aldridge", "Pau Gasol", "Danny Green",
            "Manu Ginobili", "Patty Mills", "Jonathon Simmons"
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
        "Miami Heat", "New York Knicks", "Orlando Magic", "Philadelphia 76ers", "Brooklyn Nets"
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
        ["Miami Heat", "New York Knicks", "Orlando Magic", "Philadelphia 76ers", "Brooklyn Nets"]
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
            team_idx = -1
            team_division = None
            for division in divisions:
                if team in division:
                    team_division = division
                    break
            
            if team_division is None:
                continue  # just in case

            # Assign 4-game matchups within the same division
            for team_a in team_division:
                if team_a == team:
                    team_idx = team_division.index(team_a)
                    continue  # skip self
                add_matchup(team, team_a, 4)

            # For team NOT in divisions: 
            for division in divisions:
                if team not in division:
                    for i in range(5):
                        if i < 3:
                            add_matchup(division[(team_idx + i) % 5], team, 4)
                        else:
                            add_matchup(division[(team_idx + i) % 5], team, 3)
    

    #Schedule now correctly contains every matchup each team has with opposing teams correctly!
    arrange_team(west_teams,west_divisions)
    arrange_team(east_teams,east_divisions)

    for team_a, opponents in schedule.items():
        for team_b, games in opponents.items():
            # To avoid simulating the same game twice, only simulate if team_a < team_b alphabetically
            if team_a < team_b:
                for game_num in range(1, games + 1):
                    
                    #Add players into team:
                    team_1 = 
                    




            








    


