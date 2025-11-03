"""
Test if sportsbookreview scraper can pull 2022-2024 data
"""
import sys
import json
from pathlib import Path

# Add parent directory to path to import sportsbookreview
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Create minimal config if needed
config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

# Create minimal translated.json for NFL
translated_config = {
    "nfl": {
        # Add minimal team mappings - scraper will use these to translate team names
        # Empty dict means it will just use the names as-is
    }
}

config_file = config_dir / "translated.json"
if not config_file.exists():
    with open(config_file, 'w') as f:
        json.dump(translated_config, f, indent=2)
    print(f"Created {config_file}")

# Now try importing the scraper
try:
    from sportsbookreview import NFLOddsScraper

    print("\n=== Testing Sportsbookreview Scraper ===\n")

    # Test one year at a time to see where it fails
    for year in [2022, 2023, 2024]:
        print(f"Attempting to scrape {year}...")
        try:
            scraper = NFLOddsScraper(years=[year])

            # Construct the URL it will try to fetch
            season_str = scraper._make_season(year)
            url = scraper.base + season_str
            print(f"  URL: {url}")

            # Try to fetch the data
            data = scraper.driver()

            if len(data) > 0:
                print(f"  ✅ SUCCESS: Got {len(data)} games")
                print(f"  Sample game:")
                sample = data.iloc[0]
                print(f"    Date: {sample['date']}")
                print(f"    {sample['away_team']} @ {sample['home_team']}")
                print(f"    Spread: {sample['home_open_spread']} / {sample['home_close_spread']}")
            else:
                print(f"  ⚠️  WARNING: Scraper ran but returned 0 games")

        except Exception as e:
            print(f"  ❌ FAILED: {type(e).__name__}: {str(e)}")

        print()

    print("=== Test Complete ===")

except ImportError as e:
    print(f"Failed to import scraper: {e}")
    print("\nMake sure sportsbookreview.py is in the project root")