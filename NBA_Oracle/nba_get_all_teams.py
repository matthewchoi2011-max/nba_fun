import json
import time
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import commonteamroster

# --- Configuration ---
seasons = []  # add as many seasons as needed
for year in range(1946, 2025):
    start_year = year
    end_year = year + 1
    season_str = f"{start_year}-{str(end_year)[-2:]}"
    seasons.append(season_str)
output_file = "nba_team_rosters.json"

team_rosters = {}

# Get all NBA teams
nba_teams = teams.get_teams()  # list of dicts with 'id', 'full_name', etc.

for season in seasons:
    print(f"Processing season: {season}")
    team_rosters[season] = {}
    
    for team in nba_teams:
        team_id = team["id"]
        team_name = team["full_name"]
        print(f"  Fetching roster for {team_name}...")
        
        try:
            roster_data = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
            roster_df = roster_data.get_data_frames()[0]
            
            # Extract player info
            roster = []
            for _, row in roster_df.iterrows():
                roster.append({
                    "player_id": row["PLAYER_ID"],
                    "full_name": row["PLAYER"],
                    "position": row["POSITION"],
                    "height": row["HEIGHT"],
                    "weight": row["WEIGHT"],
                    "birth_date": row["BIRTH_DATE"]
                })
            
            team_rosters[season][team_name] = roster
            time.sleep(0.6)  # avoid rate limiting
        except Exception as e:
            print(f"    Error fetching {team_name} roster for {season}: {e}")
            team_rosters[season][team_name] = []

# Save JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(team_rosters, f, indent=4, ensure_ascii=False)

print(f"\nDone! JSON saved as {output_file}")
