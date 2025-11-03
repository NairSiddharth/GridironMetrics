"""
Test The Odds API for NFL player props
"""
import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv('ODDS_API')

if not API_KEY:
    print("ERROR: ODDS_API key not found in .env file")
    exit(1)

print(f"Using API key: {API_KEY[:10]}...")

# Configuration
SPORT = 'americanfootball_nfl'
BOOKMAKERS = 'prizepicks,underdog'  # DFS sites that have player props
MARKETS = 'player_pass_tds,player_pass_yds,player_rush_yds,player_receptions'  # Player prop markets
ODDS_FORMAT = 'american'

print(f"\n=== Testing The Odds API for NFL Player Props ===")
print(f"Sport: {SPORT}")
print(f"Bookmakers: {BOOKMAKERS}")
print(f"Markets: {MARKETS}\n")

# Step 1: Get list of NFL events first
print("1. Fetching NFL events...")
events_response = requests.get(
    f'https://api.the-odds-api.com/v4/sports/{SPORT}/events',
    params={'api_key': API_KEY}
)

if events_response.status_code != 200:
    print(f'FAILED to get events: {events_response.status_code}')
    print(f'Response: {events_response.text}')
    exit(1)

events = events_response.json()
print(f'Found {len(events)} upcoming NFL events')

if len(events) == 0:
    print("No upcoming events found")
    exit(0)

# Step 2: Get player props for first event
event = events[0]
event_id = event['id']
print(f'\n2. Fetching player props for: {event["away_team"]} @ {event["home_team"]}')
print(f'   Event ID: {event_id}')

response = requests.get(
    f'https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds',
    params={
        'api_key': API_KEY,
        'bookmakers': BOOKMAKERS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
    }
)

if response.status_code != 200:
    print(f'FAILED: status {response.status_code}')
    print(f'Response: {response.text}')
else:
    data = response.json()
    print(f'SUCCESS: Found {len(data)} games with player props')

    # Show API usage
    print(f'\nRemaining requests: {response.headers.get("x-requests-remaining")}')
    print(f'Used requests: {response.headers.get("x-requests-used")}')

    # Save raw response
    output_dir = Path('research/betting_data_research')
    output_file = output_dir / 'player_props_test.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\nSaved response to: {output_file}')

    # Analyze structure
    if len(data) > 0:
        print(f'\n=== Sample Game Structure ===')
        sample = data[0]
        print(f'Game: {sample.get("away_team")} @ {sample.get("home_team")}')
        print(f'Commence: {sample.get("commence_time")}')
        print(f'Bookmakers: {len(sample.get("bookmakers", []))}')

        if len(sample.get('bookmakers', [])) > 0:
            bookmaker = sample['bookmakers'][0]
            print(f'\nBookmaker: {bookmaker.get("key")}')
            markets = bookmaker.get('markets', [])
            print(f'Markets: {len(markets)}')

            # Show player props
            for market in markets:
                market_key = market.get('key')
                print(f'\n  Market: {market_key}')
                outcomes = market.get('outcomes', [])
                print(f'  Props: {len(outcomes)}')

                # Show first few props
                for outcome in outcomes[:5]:
                    player = outcome.get('description', 'Unknown')
                    name = outcome.get('name')
                    price = outcome.get('price')
                    point = outcome.get('point', 'N/A')
                    print(f'    {player}: {name} {point} ({price})')

                if len(outcomes) > 5:
                    print(f'    ... and {len(outcomes) - 5} more props')

print('\n=== Test Complete ===')
