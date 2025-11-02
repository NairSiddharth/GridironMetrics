"""
Explore NFLWeather.com to find the correct URL structure
"""

import requests
from bs4 import BeautifulSoup
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
}

print("Exploring NFLWeather.com structure...\n")

# Try homepage
print("1. Fetching homepage...")
response = requests.get("https://nflweather.com/", headers=headers, timeout=10)
print(f"   Status: {response.status_code}")

soup = BeautifulSoup(response.content, 'html.parser')

# Find all links
all_links = soup.find_all('a', href=True)
game_links = [link['href'] for link in all_links if 'game' in link['href'].lower() or 'week' in link['href'].lower()]

print(f"\n2. Found {len(game_links)} potential game/week links")
if game_links:
    print("   Sample links:")
    for link in game_links[:10]:
        print(f"     {link}")

# Look for any year/season links
season_links = [link['href'] for link in all_links if re.search(r'202[0-9]', link['href'])]
print(f"\n3. Found {len(season_links)} links with years")
if season_links:
    print("   Sample links:")
    for link in season_links[:10]:
        print(f"     {link}")

# Save homepage for inspection
with open('nflweather_homepage.html', 'w', encoding='utf-8') as f:
    f.write(response.text)
print("\n4. Saved homepage HTML to nflweather_homepage.html")

# Try a few common URL patterns
test_urls = [
    "https://nflweather.com/week/2021/1",
    "https://nflweather.com/2021/week-1",
    "https://nflweather.com/games/2021/week/1",
]

print("\n5. Testing common URL patterns:")
for url in test_urls:
    try:
        r = requests.get(url, headers=headers, timeout=5)
        print(f"   {url} -> {r.status_code}")
        if r.status_code == 200:
            print(f"      SUCCESS! Found valid URL pattern")
            with open('nflweather_valid_page.html', 'w', encoding='utf-8') as f:
                f.write(r.text)
    except Exception as e:
        print(f"   {url} -> ERROR: {e}")

print("\nDone!")
