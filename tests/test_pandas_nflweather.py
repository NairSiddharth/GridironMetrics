"""
Test using pandas read_html to scrape NFLWeather.com tables
"""

import pandas as pd

def test_pandas_scrape():
    """Test scraping NFLWeather using pandas read_html"""

    # Test with 2021 week 1
    url = "https://nflweather.com/week/2021/1"

    print(f"Testing pandas read_html on: {url}\n")

    try:
        # Read all tables from the page
        tables = pd.read_html(url)

        print(f"[OK] Found {len(tables)} tables on the page\n")

        # Show structure of each table
        for i, table in enumerate(tables):
            print(f"=== Table {i} ===")
            print(f"Shape: {table.shape}")
            print(f"Columns: {list(table.columns)}")
            print(f"\nFirst 3 rows:")
            print(table.head(3))
            print("\n")

            # Check if this looks like a weather table
            columns_str = str(table.columns).lower()
            if any(keyword in columns_str for keyword in ['weather', 'temp', 'wind', 'forecast']):
                print(f"[OK] Table {i} appears to be weather data!")

        return len(tables) > 0

    except Exception as e:
        print(f"[ERROR] Failed to scrape tables: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("NFLWeather.com pandas read_html Test")
    print("=" * 70)

    success = test_pandas_scrape()

    print("\n" + "=" * 70)
    if success:
        print("SUCCESS: Can scrape weather tables using pandas")
        print("Ready to proceed with full implementation")
    else:
        print("FAILED: Cannot scrape tables from NFLWeather.com")
    print("=" * 70)
