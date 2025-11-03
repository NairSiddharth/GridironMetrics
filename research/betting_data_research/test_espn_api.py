"""
Test ESPN betting API endpoints for historical NFL odds data
"""
import requests
import json
from datetime import datetime

print("=== Testing ESPN Betting API ===\n")

# Test 1: Current odds
print("1. Testing current NFL odds...")
try:
    url = "https://site.web.api.espn.com/apis/v3/sports/football/nfl/odds"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"   SUCCESS: {response.status_code}")
        print(f"   Found {len(data.get('items', []))} games")

        # Save sample
        with open('research/betting_data_research/espn_current_odds_sample.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"   Saved to: espn_current_odds_sample.json")

        if len(data.get('items', [])) > 0:
            sample = data['items'][0]
            print(f"\n   Sample game:")
            print(f"   - ID: {sample.get('id')}")
            print(f"   - Teams: {sample.get('name')}")
            print(f"   - Start: {sample.get('date')}")
    else:
        print(f"   FAILED: {response.status_code}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 2: Historical team odds (2022-2024)
print("\n2. Testing historical team odds...")

# Try a few teams and years
teams_to_test = [
    ('kansas-city-chiefs', 12),  # Chiefs
    ('buffalo-bills', 2),         # Bills
]

for team_name, team_id in teams_to_test:
    print(f"\n   Team: {team_name} (ID: {team_id})")

    # Try provider ID 45 (consensus/default)
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/odds/45/past-performances?limit=200"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print(f"     SUCCESS: Got {data.get('count', 0)} historical games")

            if data.get('items'):
                # Check date range
                items = data['items']
                print(f"     Total games: {len(items)}")

                # Sample first game
                first_game_url = items[0].get('$ref') if items else None
                if first_game_url:
                    game_response = requests.get(first_game_url)
                    if game_response.status_code == 200:
                        game_data = game_response.json()
                        print(f"     Sample game data keys: {list(game_data.keys())}")

                        # Save sample
                        filename = f'research/betting_data_research/espn_historical_{team_name}_sample.json'
                        with open(filename, 'w') as f:
                            json.dump(game_data, f, indent=2)
                        print(f"     Saved sample to: {filename}")

                        # Extract useful info
                        if 'spread' in game_data:
                            print(f"     - Spread: {game_data['spread']}")
                        if 'overUnder' in game_data:
                            print(f"     - Over/Under: {game_data['overUnder']}")
                        if 'gameDate' in game_data:
                            print(f"     - Date: {game_data['gameDate']}")
        else:
            print(f"     FAILED: {response.status_code}")
    except Exception as e:
        print(f"     ERROR: {e}")

    # Just test one team to avoid too many requests
    break

# Test 3: Season futures
print("\n3. Testing season futures...")
for year in [2022, 2023, 2024]:
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{year}/futures"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"   {year}: SUCCESS - {data.get('count', 0)} futures")
        else:
            print(f"   {year}: FAILED - {response.status_code}")
    except Exception as e:
        print(f"   {year}: ERROR - {e}")

print("\n=== Test Complete ===")