"""
Test using Selenium to scrape NFLWeather.com with JavaScript rendering
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re

def test_selenium_scrape():
    """Test scraping NFLWeather using Selenium"""

    # Test with 2021 week 1
    url = "https://nflweather.com/week/2021/1"

    print(f"Testing Selenium scrape on: {url}\n")

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = None

    try:
        # Initialize Chrome driver
        print("[INFO] Initializing Chrome driver...")
        driver = webdriver.Chrome(options=chrome_options)

        # Load the page
        print(f"[INFO] Loading page: {url}")
        driver.get(url)

        # Wait for page to load (wait for body or specific element)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Give JavaScript time to render
        time.sleep(3)

        print("[OK] Page loaded successfully\n")

        # Get page source after JavaScript execution
        page_source = driver.page_source

        # Search for temperature and wind patterns
        temp_matches = re.findall(r'(\d+)°F', page_source)
        wind_matches = re.findall(r'(\d+)\s*mph', page_source, re.I)

        print(f"=== Weather Data Found ===")
        print(f"Temperature values: {temp_matches[:10]}")  # Show first 10
        print(f"Wind values: {wind_matches[:10]}")  # Show first 10

        # Try to find specific weather elements by class/id
        try:
            # Look for common weather containers
            weather_elements = driver.find_elements(By.CLASS_NAME, "weather")
            if weather_elements:
                print(f"\n[OK] Found {len(weather_elements)} elements with class 'weather'")
                for i, elem in enumerate(weather_elements[:3]):
                    print(f"  Element {i}: {elem.text[:100]}")
        except:
            pass

        # Try to find game cards or rows
        try:
            # Common patterns for game data
            game_elements = driver.find_elements(By.CLASS_NAME, "game")
            if not game_elements:
                game_elements = driver.find_elements(By.CLASS_NAME, "row")

            if game_elements:
                print(f"\n[OK] Found {len(game_elements)} game elements")
                for i, elem in enumerate(game_elements[:3]):
                    text = elem.text
                    if text and len(text) > 10:  # Filter out empty elements
                        print(f"\n  Game {i}:")
                        print(f"  {text[:200]}")
        except Exception as e:
            print(f"[WARNING] Error finding game elements: {e}")

        # Check if we found any weather data
        if temp_matches or wind_matches:
            print("\n[OK] Successfully found weather data patterns!")
            return True
        else:
            print("\n[WARNING] No weather data found after JavaScript rendering")

            # Save HTML for debugging
            with open('nflweather_selenium_debug.html', 'w', encoding='utf-8') as f:
                f.write(page_source)
            print("[INFO] Saved rendered HTML to nflweather_selenium_debug.html")

            return False

    except Exception as e:
        print(f"[ERROR] Selenium scraping failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        if driver:
            driver.quit()
            print("\n[INFO] Chrome driver closed")

if __name__ == "__main__":
    print("=" * 70)
    print("NFLWeather.com Selenium Scraping Test")
    print("=" * 70)

    success = test_selenium_scrape()

    print("\n" + "=" * 70)
    if success:
        print("SUCCESS: Can scrape weather from NFLWeather using Selenium")
        print("Ready to proceed with full implementation")
    else:
        print("FAILED: Cannot scrape weather from NFLWeather")
    print("=" * 70)
