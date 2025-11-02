"""
Weather Data Enrichment Module

Fills missing weather data in PBP cache using GitHub WeatherData repository
(2000-2020 coverage from MeteoStat and Weather Underground)
"""

import polars as pl
from pathlib import Path
from modules.logger import get_logger
from modules.constants import CACHE_DIR

logger = get_logger(__name__)

def load_github_weather_data():
    """Load weather data from GitHub CSV files"""
    logger.info("Loading GitHub weather data...")

    # Load games and weather data from cache
    weather_data_dir = Path(CACHE_DIR) / "weather_data"
    games = pl.read_csv(weather_data_dir / 'games.csv')
    weather = pl.read_csv(weather_data_dir / 'games_weather.csv', ignore_errors=True)

    logger.info(f"Loaded {len(games)} games and {len(weather)} weather measurements")

    # Aggregate hourly weather to game-level averages
    # Group by game_id and calculate mean values
    game_weather = weather.group_by('game_id').agg([
        pl.col('Temperature').mean().alias('temp'),
        pl.col('WindSpeed').mean().alias('wind'),
        pl.col('Humidity').mean().alias('humidity'),
        pl.col('Precipitation').sum().cast(pl.Float64).alias('precipitation'),
        pl.col('DewPoint').mean().alias('dew_point'),
        pl.col('Pressure').mean().alias('pressure'),
        pl.col('EstimatedCondition').first().alias('weather')  # Take first condition
    ])

    # Join with games to get season info
    game_weather = game_weather.join(games, on='game_id', how='left')

    logger.info(f"Aggregated to {len(game_weather)} games with average weather")

    return game_weather

def enrich_pbp_cache_weather(year: int, github_weather: pl.DataFrame, dry_run=False):
    """
    Enrich PBP cache with GitHub weather data for a specific year

    Args:
        year: Season year to process
        github_weather: DataFrame with GitHub weather data
        dry_run: If True, just report what would be filled without saving

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

    # Filter GitHub data for this year
    year_weather = github_weather.filter(pl.col('Season') == year)

    if len(year_weather) == 0:
        logger.warning(f"  No GitHub weather data for {year}")
        return 0, pbp['game_id'].n_unique()

    # Get unique games in PBP with null weather
    null_games = pbp.filter(
        pl.col('temp').is_null() | pl.col('wind').is_null()
    ).select(['game_id', 'game_date', 'home_team', 'away_team']).unique()

    logger.info(f"  Found {len(null_games)} unique games with null weather in PBP")

    # Convert game_date to match GitHub game_id format (YYYYMMDD)
    # game_date format is like "2000-09-03"
    null_games = null_games.with_columns([
        (pl.col('game_date').str.replace_all('-', '')).alias('date_str')
    ])

    # Match to GitHub data by creating date-based lookup
    year_weather = year_weather.with_columns([
        (pl.col('game_id').cast(pl.String).str.slice(0, 8)).alias('date_str')
    ])

    enriched_count = 0

    # For each null game, try to find matching GitHub weather
    for row in null_games.iter_rows(named=True):
        game_id = row['game_id']
        date_str = row['date_str']

        # Find matching weather by date
        matching_weather = year_weather.filter(pl.col('date_str') == date_str)

        if len(matching_weather) == 0:
            continue

        # If multiple games on same date, use first one (could improve matching with stadium)
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

def enrich_all_years(start_year=2000, end_year=2020, dry_run=False):
    """Enrich PBP cache for all years with GitHub weather data"""

    # Load GitHub weather data once
    github_weather = load_github_weather_data()

    total_enriched = 0
    total_games = 0

    for year in range(start_year, end_year + 1):
        enriched, games = enrich_pbp_cache_weather(year, github_weather, dry_run=dry_run)
        total_enriched += enriched
        total_games += games

    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Total games enriched: {total_enriched}")
    logger.info(f"Total games processed: {total_games}")
    logger.info(f"Enrichment rate: {total_enriched/total_games*100:.1f}%")

if __name__ == "__main__":
    import sys

    # Test on single year first
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        github_weather = load_github_weather_data()
        enrich_pbp_cache_weather(year, github_weather, dry_run=False)
    else:
        # Run full enrichment
        enrich_all_years(dry_run=False)
