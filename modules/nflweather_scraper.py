"""
NFLWeather.com Scraper Module

Scrapes weather data from NFLWeather.com for 2021-2025 seasons
and fills missing weather data in PBP cache.
"""

import polars as pl
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
from modules.logger import get_logger
from modules.constants import CACHE_DIR

logger = get_logger(__name__)

# Team name mapping (NFLWeather uses full names, need to map to abbreviations)
TEAM_NAME_MAP = {
    'cardinals': 'ARI', 'falcons': 'ATL', 'ravens': 'BAL', 'bills': 'BUF',
    'panthers': 'CAR', 'bears': 'CHI', 'bengals': 'CIN', 'browns': 'CLE',
    'cowboys': 'DAL', 'broncos': 'DEN', 'lions': 'DET', 'packers': 'GB',
    'texans': 'HOU', 'colts': 'IND', 'jaguars': 'JAX', 'chiefs': 'KC',
    'raiders': 'LV', 'chargers': 'LAC', 'rams': 'LAR', 'dolphins': 'MIA',
    'vikings': 'MIN', 'patriots': 'NE', 'saints': 'NO', 'giants': 'NYG',
    'jets': 'NYJ', 'eagles': 'PHI', 'steelers': 'PIT', 'seahawks': 'SEA',
    'ers': 'SF', '49ers': 'SF', 'buccaneers': 'TB', 'titans': 'TEN',
    'washington': 'WAS', 'commanders': 'WAS', 'redskins': 'WAS'
}


def init_driver():
    """Initialize Chrome driver with headless options"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    return webdriver.Chrome(options=chrome_options)


def parse_team_from_url(team_text):
    """Parse team abbreviation from NFLWeather team name"""
    team_lower = team_text.lower().strip()

    for key, abbrev in TEAM_NAME_MAP.items():
        if key in team_lower:
            return abbrev

    logger.warning(f"Could not map team: {team_text}")
    return None


def scrape_week(driver, year, week):
    """
    Scrape weather data for a specific week

    Args:
        driver: Selenium WebDriver instance
        year: Season year
        week: Week number

    Returns:
        List of dicts with game weather data
    """
    url = f"https://www.nflweather.com/week/{year}/week-{week}"

    logger.info(f"  Scraping {year} Week {week}...")

    try:
        driver.get(url)

        # Wait for page load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Give JavaScript time to render
        time.sleep(2)

        games = []
        seen_games = set()

        # Find all individual game cards (they're in col-* Bootstrap columns)
        game_cards = driver.find_elements(By.CSS_SELECTOR, "div[class*='col-']")

        for card in game_cards:
            try:
                # Find game link within this card
                try:
                    link = card.find_element(By.CSS_SELECTOR, "a[href*='/games/']")
                except:
                    continue  # Not a game card

                href = link.get_attribute('href')

                # Extract teams from URL
                # Format: /games/2021/week-1/away-at-home
                match = re.search(r'/games/\d{4}/week-\d+/([^/]+)-at-([^/]+)', href)
                if not match:
                    continue

                away_text, home_text = match.groups()
                away_team = parse_team_from_url(away_text)
                home_team = parse_team_from_url(home_text)

                if not away_team or not home_team:
                    continue

                # Skip duplicates
                game_key = (away_team, home_team)
                if game_key in seen_games:
                    continue
                seen_games.add(game_key)

                # Get weather data from THIS card only
                weather_text = card.text

                # Parse temperature
                temp_match = re.search(r'(\d+)\s*°?F', weather_text)
                temp = float(temp_match.group(1)) if temp_match else None

                # Parse wind
                wind_match = re.search(r'(\d+)\s*mph', weather_text, re.I)
                wind = float(wind_match.group(1)) if wind_match else None

                # Parse weather description (look for common patterns)
                weather_desc = None
                for pattern in ['Clear', 'Cloudy', 'Rain', 'Snow', 'Partly Cloudy', 'Overcast', 'Showers']:
                    if pattern.lower() in weather_text.lower():
                        weather_desc = pattern
                        break

                games.append({
                    'year': year,
                    'week': week,
                    'away_team': away_team,
                    'home_team': home_team,
                    'temp': temp,
                    'wind': wind,
                    'weather': weather_desc
                })

            except Exception as e:
                logger.debug(f"    Error parsing game card: {e}")
                continue

        logger.info(f"    Found {len(games)} games")
        return games

    except Exception as e:
        logger.error(f"  Error scraping week {week}: {e}")
        return []


def scrape_season(year, max_week=18):
    """
    Scrape weather data for an entire season

    Args:
        year: Season year
        max_week: Maximum week number (default 18 for regular season)

    Returns:
        DataFrame with weather data
    """
    logger.info(f"Scraping NFLWeather for {year} season...")

    driver = None
    all_games = []

    try:
        driver = init_driver()

        for week in range(1, max_week + 1):
            week_games = scrape_week(driver, year, week)
            all_games.extend(week_games)

            # Rate limiting - be respectful
            time.sleep(2)

        if len(all_games) == 0:
            logger.warning(f"No games found for {year}")
            return pl.DataFrame()

        return pl.DataFrame(all_games)

    except Exception as e:
        logger.error(f"Error scraping season {year}: {e}")
        return pl.DataFrame()

    finally:
        if driver:
            driver.quit()


def enrich_pbp_with_nflweather(year, nflweather_data, dry_run=False):
    """
    Enrich PBP cache with NFLWeather data for a specific year

    Args:
        year: Season year
        nflweather_data: DataFrame with NFLWeather data
        dry_run: If True, just report what would be filled

    Returns:
        Tuple of (games_enriched, games_total)
    """
    pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{year}.parquet"

    if not pbp_file.exists():
        logger.warning(f"PBP file not found: {pbp_file}")
        return 0, 0

    logger.info(f"Processing {year}...")

    # Load PBP data
    pbp = pl.read_parquet(pbp_file)

    # Count nulls before
    null_temp_before = pbp.filter(pl.col('temp').is_null())['game_id'].n_unique()
    null_wind_before = pbp.filter(pl.col('wind').is_null())['game_id'].n_unique()

    logger.info(f"  Before: {null_temp_before} games with null temp, {null_wind_before} with null wind")

    # Filter NFLWeather data for this year
    year_weather = nflweather_data.filter(pl.col('year') == year)

    if len(year_weather) == 0:
        logger.warning(f"  No NFLWeather data for {year}")
        return 0, pbp['game_id'].n_unique()

    # Get unique games with null weather
    null_games = pbp.filter(
        pl.col('temp').is_null() | pl.col('wind').is_null()
    ).select(['game_id', 'week', 'home_team', 'away_team']).unique()

    logger.info(f"  Found {len(null_games)} unique games with null weather in PBP")

    enriched_count = 0

    # For each null game, try to find matching NFLWeather data
    for row in null_games.iter_rows(named=True):
        game_id = row['game_id']
        week = row['week']
        home_team = row['home_team']
        away_team = row['away_team']

        # Find matching weather by week and teams
        matching_weather = year_weather.filter(
            (pl.col('week') == week) &
            (pl.col('home_team') == home_team) &
            (pl.col('away_team') == away_team)
        )

        if len(matching_weather) == 0:
            continue

        weather_row = matching_weather[0]

        new_temp = weather_row['temp']
        new_wind = weather_row['wind']
        new_weather_desc = weather_row['weather']

        if not dry_run:
            # Update PBP rows for this game
            pbp = pbp.with_columns([
                pl.when(pl.col('game_id') == game_id)
                  .then(pl.coalesce(pl.col('temp'), pl.lit(new_temp)))
                  .otherwise(pl.col('temp'))
                  .alias('temp'),

                pl.when(pl.col('game_id') == game_id)
                  .then(pl.coalesce(pl.col('wind'), pl.lit(new_wind)))
                  .otherwise(pl.col('wind'))
                  .alias('wind'),

                pl.when(pl.col('game_id') == game_id)
                  .then(pl.coalesce(pl.col('weather'), pl.lit(new_weather_desc)))
                  .otherwise(pl.col('weather'))
                  .alias('weather'),
            ])

        enriched_count += 1

    # Count nulls after
    null_temp_after = pbp.filter(pl.col('temp').is_null())['game_id'].n_unique()
    null_wind_after = pbp.filter(pl.col('wind').is_null())['game_id'].n_unique()

    logger.info(f"  After: {null_temp_after} games with null temp, {null_wind_after} with null wind")
    logger.info(f"  Enriched {enriched_count} games")

    if not dry_run and enriched_count > 0:
        # Save updated PBP
        pbp.write_parquet(pbp_file)
        logger.info(f"  Saved updated PBP cache for {year}")

    return enriched_count, pbp['game_id'].n_unique()


def enrich_all_years(start_year=2021, end_year=2025, dry_run=False):
    """
    Scrape NFLWeather and enrich PBP cache for multiple years

    Args:
        start_year: First year to process
        end_year: Last year to process
        dry_run: If True, just report what would be filled
    """
    logger.info(f"=== NFLWeather Enrichment: {start_year}-{end_year} ===")

    total_enriched = 0
    total_games = 0

    for year in range(start_year, end_year + 1):
        # Scrape NFLWeather for this year
        nflweather_data = scrape_season(year)

        if len(nflweather_data) == 0:
            logger.warning(f"No weather data scraped for {year}, skipping")
            continue

        # Enrich PBP cache
        enriched, games = enrich_pbp_with_nflweather(year, nflweather_data, dry_run=dry_run)
        total_enriched += enriched
        total_games += games

        # Rate limiting between years
        time.sleep(3)

    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Total games enriched: {total_enriched}")
    logger.info(f"Total games processed: {total_games}")
    if total_games > 0:
        logger.info(f"Enrichment rate: {total_enriched/total_games*100:.1f}%")


if __name__ == "__main__":
    import sys

    # Test on single year first
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        weather_data = scrape_season(year)
        logger.info(f"\nScraped {len(weather_data)} games for {year}")
        if len(weather_data) > 0:
            logger.info(f"Sample data:\n{weather_data.head(10)}")
            enrich_pbp_with_nflweather(year, weather_data, dry_run=False)
    else:
        # Run full enrichment
        enrich_all_years(dry_run=False)
