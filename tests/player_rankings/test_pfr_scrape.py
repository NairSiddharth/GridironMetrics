"""
Test script to verify we can scrape weather data from Pro-Football-Reference
"""

import requests
from bs4 import BeautifulSoup
import time

def test_scrape_pfr_weather():
    """Test scraping weather from a single PFR game"""

    # Test with the game you provided
    url = "https://www.pro-football-reference.com/boxscores/200009030nwe.htm"

    print(f"Testing PFR scrape: {url}")

    # Use session for better cookie handling
    session = requests.Session()

    # Set proper headers to avoid 403 (more realistic browser simulation)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.pro-football-reference.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin'
    }

    try:
        # First visit the homepage to get cookies
        print("Visiting homepage first to get cookies...")
        session.get('https://www.pro-football-reference.com/', headers=headers)
        time.sleep(1)

        # Now request the boxscore page
        response = session.get(url, headers=headers)
        response.raise_for_status()

        print(f"[OK] Successfully fetched page (status: {response.status_code})")

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find weather data using the selector you provided
        weather_row = soup.find('th', {'data-stat': 'info', 'scope': 'row'}, string='Weather')

        if weather_row:
            # Get the value in the next <td>
            weather_value = weather_row.find_next_sibling('td')
            if weather_value:
                weather_text = weather_value.get_text(strip=True)
                print(f"[OK] Found weather data: '{weather_text}'")

                # Try to parse temperature and wind
                # Example format: "72 degrees, relative humidity 54%, wind 6 mph"
                import re

                temp_match = re.search(r'(\d+)\s*degrees?', weather_text)
                wind_match = re.search(r'wind\s+(\d+)\s*mph', weather_text)
                humidity_match = re.search(r'humidity\s+(\d+)%', weather_text)

                print("\nParsed values:")
                print(f"  Temperature: {temp_match.group(1) if temp_match else 'N/A'}°F")
                print(f"  Wind: {wind_match.group(1) if wind_match else 'N/A'} mph")
                print(f"  Humidity: {humidity_match.group(1) if humidity_match else 'N/A'}%")

                return True
            else:
                print("[ERROR] Found weather label but no value")
                return False
        else:
            print("[ERROR] Weather data not found on page")

            # Debug: show all th elements with data-stat="info"
            print("\nAll 'info' rows found:")
            for th in soup.find_all('th', {'data-stat': 'info'}):
                value = th.find_next_sibling('td')
                print(f"  {th.get_text(strip=True)}: {value.get_text(strip=True) if value else 'N/A'}")

            return False

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error fetching page: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PFR Weather Scraping Test")
    print("=" * 60)

    success = test_scrape_pfr_weather()

    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: Can scrape weather from PFR")
        print("Ready to proceed with full implementation")
    else:
        print("FAILED: Cannot scrape weather from PFR")
        print("May need alternative approach")
    print("=" * 60)
