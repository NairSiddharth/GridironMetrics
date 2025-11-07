"""
Test script to verify we can scrape weather data from NFLWeather.com
"""

import requests
from bs4 import BeautifulSoup
import time
import re

def test_scrape_nflweather():
    """Test scraping weather from NFLWeather.com"""

    # Test with a recent season's game
    # NFLWeather URL format: https://nflweather.com/games/YEAR/week-N/team-at-team
    # Let's try week 1 of 2021 season

    test_url = "https://nflweather.com/games/2021/week-1/buccaneers-at-cowboys"

    print(f"Testing NFLWeather scrape: {test_url}")

    # Use session for better cookie handling
    session = requests.Session()

    # Set proper headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    try:
        # Request the game page
        print("Fetching NFLWeather page...")
        response = session.get(test_url, headers=headers, timeout=10)
        response.raise_for_status()

        print(f"[OK] Successfully fetched page (status: {response.status_code})")

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Try to find weather data
        # NFLWeather typically has weather info in specific div/span elements
        # Let's inspect the structure

        # Look for temperature
        temp_elem = soup.find('span', class_=re.compile(r'temp|temperature', re.I))
        if not temp_elem:
            # Try alternative selectors
            temp_elem = soup.find(string=re.compile(r'\d+°F'))

        # Look for wind
        wind_elem = soup.find(string=re.compile(r'wind.*\d+.*mph', re.I))

        # Look for humidity
        humidity_elem = soup.find(string=re.compile(r'humidity.*\d+%', re.I))

        # Print raw HTML snippet to understand structure
        print("\n=== Searching for weather patterns in HTML ===")

        # Search for temperature patterns
        temp_matches = re.findall(r'(\d+)°F', response.text)
        if temp_matches:
            print(f"Found temperature values: {temp_matches}")

        # Search for wind patterns
        wind_matches = re.findall(r'wind[:\s]+(\d+)\s*mph', response.text, re.I)
        if wind_matches:
            print(f"Found wind values: {wind_matches}")

        # Search for humidity patterns
        humidity_matches = re.findall(r'humidity[:\s]+(\d+)%', response.text, re.I)
        if humidity_matches:
            print(f"Found humidity values: {humidity_matches}")

        # Look for common weather-related class names or IDs
        print("\n=== Weather-related elements ===")
        weather_divs = soup.find_all(['div', 'span', 'p'], class_=re.compile(r'weather|temp|wind|humidity', re.I))
        for elem in weather_divs[:5]:  # Show first 5
            print(f"{elem.name}.{elem.get('class', [])}: {elem.get_text(strip=True)[:100]}")

        # Check if we found any weather data
        if temp_matches or wind_matches or humidity_matches:
            print("\n[OK] Found weather data patterns!")
            print(f"  Temperature: {temp_matches[0] if temp_matches else 'N/A'}°F")
            print(f"  Wind: {wind_matches[0] if wind_matches else 'N/A'} mph")
            print(f"  Humidity: {humidity_matches[0] if humidity_matches else 'N/A'}%")
            return True
        else:
            print("\n[WARNING] No weather data patterns found")
            print("Saving HTML for manual inspection...")
            with open('nflweather_debug.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Saved to nflweather_debug.html")
            return False

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error fetching page: {e}")
        return False

def test_season_page():
    """Test scraping the season overview page to get game URLs"""

    # NFLWeather season page format
    season_url = "https://nflweather.com/week/2021/1"

    print(f"\n\nTesting season page: {season_url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }

    try:
        response = requests.get(season_url, headers=headers, timeout=10)
        response.raise_for_status()

        print(f"[OK] Successfully fetched season page (status: {response.status_code})")

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all game links
        game_links = soup.find_all('a', href=re.compile(r'/games/\d{4}/week-\d+/'))

        print(f"Found {len(game_links)} game links")

        if len(game_links) > 0:
            print("\nSample game URLs:")
            for link in game_links[:5]:
                href = link.get('href')
                full_url = f"https://nflweather.com{href}" if href.startswith('/') else href
                print(f"  {full_url}")
            return True
        else:
            print("[WARNING] No game links found")
            return False

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error fetching season page: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("NFLWeather.com Scraping Test")
    print("=" * 70)

    # Test individual game page
    game_success = test_scrape_nflweather()

    # Test season overview page
    season_success = test_season_page()

    print("\n" + "=" * 70)
    if game_success and season_success:
        print("SUCCESS: Can scrape weather from NFLWeather.com")
        print("Ready to proceed with full implementation")
    elif game_success:
        print("PARTIAL: Can access game pages, but need to refine data extraction")
    else:
        print("FAILED: Cannot scrape weather from NFLWeather.com")
        print("May need alternative approach or manual inspection")
    print("=" * 70)
