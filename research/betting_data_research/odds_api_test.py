"""
Test script to pull NFL odds from The Odds API
"""
import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv('ODDS_API')

if not API_KEY:
    print("ERROR: ODDS_API key not found in .env file")
    exit(1)

print(f"Using API key: {API_KEY[:10]}...")

# Configuration
SPORT = 'americanfootball_nfl'
REGIONS = 'us'  # US bookmakers
MARKETS = 'h2h,spreads,totals'  # Moneyline, spreads, totals
ODDS_FORMAT = 'american'  # American odds format
DATE_FORMAT = 'iso'

print(f"\n=== Testing The Odds API ===")
print(f"Sport: {SPORT}")
print(f"Markets: {MARKETS}")
print(f"Regions: {REGIONS}\n")

# Test 1: Get list of available sports
print("1. Fetching available sports...")
sports_response = requests.get(
    'https://api.the-odds-api.com/v4/sports',
    params={'api_key': API_KEY}
)

if sports_response.status_code != 200:
    print(f'  FAILED: status {sports_response.status_code}')
    print(f'  Response: {sports_response.text}')
else:
    sports = sports_response.json()
    nfl_sports = [s for s in sports if 'nfl' in s.get('key', '').lower()]
    print(f'  SUCCESS: Found {len(sports)} sports')
    print(f'  NFL options: {nfl_sports}')

# Test 2: Get current/upcoming NFL odds
print("\n2. Fetching current NFL odds...")
odds_response = requests.get(
    f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds',
    params={
        'api_key': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT,
    }
)

if odds_response.status_code != 200:
    print(f'  FAILED: status {odds_response.status_code}')
    print(f'  Response: {odds_response.text}')
else:
    odds_data = odds_response.json()
    print(f'  SUCCESS: Found {len(odds_data)} upcoming games')

    # Show API usage
    print(f'\n  Remaining requests: {odds_response.headers.get("x-requests-remaining")}')
    print(f'  Used requests: {odds_response.headers.get("x-requests-used")}')

    # Save raw response
    output_dir = Path('research/betting_data_research')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'odds_api_test.json'
    with open(output_file, 'w') as f:
        json.dump(odds_data, f, indent=2)
    print(f'\n  Saved response to: {output_file}')

    # Show sample game
    if len(odds_data) > 0:
        print(f'\n  Sample game structure:')
        sample = odds_data[0]
        print(f'    Game ID: {sample.get("id")}')
        print(f'    Home: {sample.get("home_team")}')
        print(f'    Away: {sample.get("away_team")}')
        print(f'    Commence: {sample.get("commence_time")}')
        print(f'    Bookmakers: {len(sample.get("bookmakers", []))}')

        if len(sample.get('bookmakers', [])) > 0:
            bookmaker = sample['bookmakers'][0]
            print(f'\n    Sample bookmaker: {bookmaker.get("key")}')
            print(f'    Markets available: {len(bookmaker.get("markets", []))}')

            for market in bookmaker.get('markets', []):
                print(f'\n    Market: {market.get("key")}')
                for outcome in market.get('outcomes', []):
                    print(f'      {outcome.get("name")}: {outcome.get("price")} (point: {outcome.get("point", "N/A")})')

print('\n=== Test Complete ===')
