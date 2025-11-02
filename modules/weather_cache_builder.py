"""
Weather Cache Builder Module

Analyzes player performance across different weather conditions and builds
cached weather adjustment factors for Phase 4.5.

Uses play-by-play data (already cached) to extract weather info and calculate
position-level performance baselines and player-specific adjustments.
"""

import polars as pl
from pathlib import Path
from modules.logger import get_logger
from modules.constants import (
    # Temperature thresholds
    WEATHER_TEMP_COLD,
    WEATHER_TEMP_COOL,
    WEATHER_TEMP_MODERATE,

    # Wind thresholds
    WEATHER_WIND_CALM,
    WEATHER_WIND_MODERATE,

    # Adjustment caps
    WEATHER_FACTOR_MIN,
    WEATHER_FACTOR_MAX,
    WEATHER_TOTAL_MIN,
    WEATHER_TOTAL_MAX,

    # Minimum sample size
    WEATHER_MIN_PLAYS,

    # Precipitation keywords
    WEATHER_PRECIP_KEYWORDS,

    CACHE_DIR
)

# Initialize logger
logger = get_logger(__name__)

# Cache directory following established pattern
WEATHER_CACHE_DIR = Path(CACHE_DIR) / "weather"
WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def categorize_temperature(temp: float) -> str:
    """
    Categorize temperature into buckets.

    Args:
        temp: Temperature in Fahrenheit

    Returns:
        Category: 'cold', 'cool', 'moderate', 'hot'
    """
    if temp is None:
        return 'unknown'

    if temp < WEATHER_TEMP_COLD:
        return 'cold'
    elif temp < WEATHER_TEMP_COOL:
        return 'cool'
    elif temp < WEATHER_TEMP_MODERATE:
        return 'moderate'
    else:
        return 'hot'


def categorize_wind(wind: float) -> str:
    """
    Categorize wind speed into buckets.

    Args:
        wind: Wind speed in mph

    Returns:
        Category: 'calm', 'moderate', 'high'
    """
    if wind is None:
        return 'unknown'

    if wind < WEATHER_WIND_CALM:
        return 'calm'
    elif wind < WEATHER_WIND_MODERATE:
        return 'moderate'
    else:
        return 'high'


def has_precipitation(weather_desc: str) -> bool:
    """
    Detect precipitation from weather description.

    Args:
        weather_desc: Weather description string (e.g., "Light Snow", "Sunny")

    Returns:
        True if precipitation detected, False otherwise
    """
    if not weather_desc or not isinstance(weather_desc, str):
        return False

    weather_lower = weather_desc.lower()
    return any(keyword in weather_lower for keyword in WEATHER_PRECIP_KEYWORDS)


def categorize_environment(roof: str) -> str:
    """
    Categorize playing environment.

    Args:
        roof: Roof type ('dome', 'outdoors', 'closed', 'open')

    Returns:
        Category: 'dome', 'outdoor'
    """
    if roof in ['dome', 'closed']:
        return 'dome'
    else:
        return 'outdoor'


def build_weather_performance_cache(
    season: int,
    position: str
) -> pl.DataFrame:
    """
    Build weather performance cache for a specific position and season.

    This is the CORE function that calculates:
    1. Position-average performance in each weather condition
    2. Individual player performance in each weather condition
    3. Adjustment factors (player_perf / position_avg)

    Args:
        season: NFL season year (e.g., 2024)
        position: Position code ('QB', 'RB', 'WR', 'TE')

    Returns:
        DataFrame with weather adjustment factors per player:
        - player_id (str)
        - player_name (str)
        - temp_cold_adj (float): 0.95-1.05
        - temp_cool_adj (float): 0.95-1.05
        - temp_moderate_adj (float): 0.95-1.05
        - temp_hot_adj (float): 0.95-1.05
        - wind_calm_adj (float): 0.95-1.05
        - wind_moderate_adj (float): 0.95-1.05
        - wind_high_adj (float): 0.95-1.05
        - precip_adj (float): 0.95-1.05
        - dome_adj (float): 0.95-1.05
        - outdoor_adj (float): 0.95-1.05
        - total_plays (int): Sample size for validation
    """
    cache_file = WEATHER_CACHE_DIR / f"weather_{position.lower()}_{season}.parquet"

    # Check if cached
    if cache_file.exists():
        logger.debug(f"Loading cached weather data for {position} {season}")
        return pl.read_parquet(cache_file)

    logger.info(f"Building weather performance cache for {position} {season}...")

    try:
        # Load play-by-play data (already cached by main.py)
        pbp_cache_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"

        if not pbp_cache_file.exists():
            logger.warning(f"PBP data not cached for {season}. Run main pipeline first.")
            return pl.DataFrame()

        pbp = pl.read_parquet(pbp_cache_file)

        # Filter to relevant plays for this position
        if position == 'QB':
            # QB passing plays
            relevant_plays = pbp.filter(
                (pl.col('play_type') == 'pass') &
                pl.col('passer_player_id').is_not_null()
            ).with_columns([
                pl.col('passer_player_id').alias('player_id'),
                pl.col('passer_player_name').alias('player_name'),
                pl.col('passing_yards').fill_null(0).alias('yards'),
                pl.col('pass_touchdown').fill_null(0).cast(pl.Int32).alias('td'),
                pl.col('complete_pass').fill_null(0).cast(pl.Int32).alias('success')
            ])

        elif position == 'RB':
            # RB rushing plays
            relevant_plays = pbp.filter(
                (pl.col('play_type') == 'run') &
                pl.col('rusher_player_id').is_not_null()
            ).with_columns([
                pl.col('rusher_player_id').alias('player_id'),
                pl.col('rusher_player_name').alias('player_name'),
                pl.col('rushing_yards').fill_null(0).alias('yards'),
                pl.col('rush_touchdown').fill_null(0).cast(pl.Int32).alias('td'),
                (pl.col('rushing_yards').fill_null(0) > 0).cast(pl.Int32).alias('success')
            ])

        elif position in ['WR', 'TE']:
            # WR/TE receiving plays
            relevant_plays = pbp.filter(
                (pl.col('play_type') == 'pass') &
                pl.col('receiver_player_id').is_not_null()
            ).with_columns([
                pl.col('receiver_player_id').alias('player_id'),
                pl.col('receiver_player_name').alias('player_name'),
                pl.col('receiving_yards').fill_null(0).alias('yards'),
                pl.col('pass_touchdown').fill_null(0).cast(pl.Int32).alias('td'),
                pl.col('complete_pass').fill_null(0).cast(pl.Int32).alias('success')
            ])

        else:
            logger.error(f"Invalid position: {position}")
            return pl.DataFrame()

        # Add weather categories
        relevant_plays = relevant_plays.with_columns([
            pl.col('temp').map_elements(categorize_temperature, return_dtype=pl.Utf8).alias('temp_cat'),
            pl.col('wind').map_elements(categorize_wind, return_dtype=pl.Utf8).alias('wind_cat'),
            pl.col('weather').map_elements(has_precipitation, return_dtype=pl.Boolean).alias('has_precip'),
            pl.col('roof').map_elements(categorize_environment, return_dtype=pl.Utf8).alias('env_cat')
        ])

        # Calculate position-level averages for each condition
        # Temperature averages
        temp_avgs = {}
        for temp_cat in ['cold', 'cool', 'moderate', 'hot']:
            temp_plays = relevant_plays.filter(pl.col('temp_cat') == temp_cat)
            if len(temp_plays) > 0:
                temp_avgs[temp_cat] = temp_plays['yards'].mean()
            else:
                temp_avgs[temp_cat] = None

        # Wind averages
        wind_avgs = {}
        for wind_cat in ['calm', 'moderate', 'high']:
            wind_plays = relevant_plays.filter(pl.col('wind_cat') == wind_cat)
            if len(wind_plays) > 0:
                wind_avgs[wind_cat] = wind_plays['yards'].mean()
            else:
                wind_avgs[wind_cat] = None

        # Precipitation averages
        precip_avg = None
        precip_plays = relevant_plays.filter(pl.col('has_precip') == True)
        if len(precip_plays) > 0:
            precip_avg = precip_plays['yards'].mean()

        clear_avg = None
        clear_plays = relevant_plays.filter(pl.col('has_precip') == False)
        if len(clear_plays) > 0:
            clear_avg = clear_plays['yards'].mean()

        # Environment averages
        dome_avg = None
        dome_plays = relevant_plays.filter(pl.col('env_cat') == 'dome')
        if len(dome_plays) > 0:
            dome_avg = dome_plays['yards'].mean()

        outdoor_avg = None
        outdoor_plays = relevant_plays.filter(pl.col('env_cat') == 'outdoor')
        if len(outdoor_plays) > 0:
            outdoor_avg = outdoor_plays['yards'].mean()

        # Calculate player-level performance in each condition
        player_weather_stats = []

        for player_id in relevant_plays['player_id'].unique():
            player_plays = relevant_plays.filter(pl.col('player_id') == player_id)
            player_name = player_plays['player_name'][0]

            # Need minimum plays
            if len(player_plays) < WEATHER_MIN_PLAYS:
                continue

            player_stats = {
                'player_id': player_id,
                'player_name': player_name,
                'total_plays': len(player_plays)
            }

            # Temperature adjustments
            for temp_cat in ['cold', 'cool', 'moderate', 'hot']:
                temp_player_plays = player_plays.filter(pl.col('temp_cat') == temp_cat)

                if len(temp_player_plays) >= WEATHER_MIN_PLAYS and temp_avgs[temp_cat] is not None and temp_avgs[temp_cat] > 0:
                    player_avg = temp_player_plays['yards'].mean()
                    adj = player_avg / temp_avgs[temp_cat]
                    adj = max(WEATHER_FACTOR_MIN, min(WEATHER_FACTOR_MAX, adj))
                else:
                    # Use position average if insufficient data
                    adj = 1.0

                player_stats[f'temp_{temp_cat}_adj'] = adj

            # Wind adjustments
            for wind_cat in ['calm', 'moderate', 'high']:
                wind_player_plays = player_plays.filter(pl.col('wind_cat') == wind_cat)

                if len(wind_player_plays) >= WEATHER_MIN_PLAYS and wind_avgs[wind_cat] is not None and wind_avgs[wind_cat] > 0:
                    player_avg = wind_player_plays['yards'].mean()
                    adj = player_avg / wind_avgs[wind_cat]
                    adj = max(WEATHER_FACTOR_MIN, min(WEATHER_FACTOR_MAX, adj))
                else:
                    adj = 1.0

                player_stats[f'wind_{wind_cat}_adj'] = adj

            # Precipitation adjustment
            precip_player_plays = player_plays.filter(pl.col('has_precip') == True)
            if len(precip_player_plays) >= WEATHER_MIN_PLAYS and precip_avg is not None and precip_avg > 0:
                player_avg = precip_player_plays['yards'].mean()
                adj = player_avg / precip_avg
                adj = max(WEATHER_FACTOR_MIN, min(WEATHER_FACTOR_MAX, adj))
            else:
                adj = 1.0
            player_stats['precip_adj'] = adj

            # Environment adjustments
            for env_cat, env_avg in [('dome', dome_avg), ('outdoor', outdoor_avg)]:
                env_player_plays = player_plays.filter(pl.col('env_cat') == env_cat)

                if len(env_player_plays) >= WEATHER_MIN_PLAYS and env_avg is not None and env_avg > 0:
                    player_avg = env_player_plays['yards'].mean()
                    adj = player_avg / env_avg
                    adj = max(WEATHER_FACTOR_MIN, min(WEATHER_FACTOR_MAX, adj))
                else:
                    adj = 1.0

                player_stats[f'{env_cat}_adj'] = adj

            player_weather_stats.append(player_stats)

        # Convert to DataFrame
        if len(player_weather_stats) == 0:
            logger.warning(f"No players with sufficient plays for {position} {season}")
            return pl.DataFrame()

        weather_df = pl.DataFrame(player_weather_stats)

        # Cache the result
        weather_df.write_parquet(cache_file)
        logger.info(f"Cached weather data for {len(weather_df)} {position}s in {season}")

        return weather_df

    except Exception as e:
        logger.error(f"Failed to build weather cache for {position} {season}: {e}")
        import traceback
        traceback.print_exc()
        return pl.DataFrame()


def calculate_weather_adjustment(
    player_id: str,
    season: int,
    position: str,
    game_temp: float,
    game_wind: float,
    game_weather: str,
    game_roof: str
) -> float:
    """
    Calculate weather adjustment for a player in a specific game.

    This is the MAIN ENTRY POINT called from Phase 4.5.

    Args:
        player_id: Player's GSIS ID
        season: Season year
        position: Position code ('QB', 'RB', 'WR', 'TE')
        game_temp: Game temperature (F)
        game_wind: Game wind speed (mph)
        game_weather: Weather description
        game_roof: Roof type

    Returns:
        Weather adjustment multiplier (0.90-1.10)
        Returns 1.0 if no data available
    """
    try:
        # Load weather performance cache
        weather_data = build_weather_performance_cache(season, position)

        if weather_data.is_empty():
            logger.debug(f"No weather data available for {position} {season}")
            return 1.0

        # Get player's weather adjustments
        player_weather = weather_data.filter(pl.col('player_id') == player_id)

        if len(player_weather) == 0:
            # Player not found (likely insufficient sample size)
            logger.debug(f"Player {player_id} not in weather cache (insufficient plays)")
            return 1.0

        # Get first row as a dictionary for scalar access
        player_row = player_weather.row(0, named=True)

        # Categorize current game conditions
        temp_cat = categorize_temperature(game_temp)
        wind_cat = categorize_wind(game_wind)
        has_precip = has_precipitation(game_weather)
        env_cat = categorize_environment(game_roof)

        # Get adjustment factors
        if temp_cat != 'unknown':
            temp_adj = player_row[f'temp_{temp_cat}_adj']
        else:
            temp_adj = 1.0

        if wind_cat != 'unknown':
            wind_adj = player_row[f'wind_{wind_cat}_adj']
        else:
            wind_adj = 1.0

        precip_adj = player_row['precip_adj'] if has_precip else 1.0
        env_adj = player_row[f'{env_cat}_adj']

        # Combine multiplicatively
        combined_adj = temp_adj * wind_adj * precip_adj * env_adj

        # Apply final cap
        final_adj = max(WEATHER_TOTAL_MIN, min(WEATHER_TOTAL_MAX, combined_adj))

        logger.debug(
            f"{player_row['player_name']}: Weather adjustment = {final_adj:.3f} "
            f"(temp={temp_adj:.3f}, wind={wind_adj:.3f}, precip={precip_adj:.3f}, env={env_adj:.3f})"
        )

        return final_adj

    except Exception as e:
        logger.error(f"Error calculating weather adjustment: {e}")
        return 1.0


def build_all_weather_caches(start_year: int = 2016, end_year: int = 2025) -> None:
    """
    Build weather performance caches for all positions and years.

    Args:
        start_year: First year to cache (default: 2016)
        end_year: Last year to cache (default: 2025)
    """
    logger.info(f"Building weather caches for {start_year}-{end_year}...")

    positions = ['QB', 'RB', 'WR', 'TE']

    for year in range(start_year, end_year + 1):
        for position in positions:
            try:
                build_weather_performance_cache(year, position)
            except Exception as e:
                logger.error(f"Failed to build weather cache for {position} {year}: {e}")
                continue

    logger.info("Weather cache building complete")


def cache_is_up_to_date(start_year: int = 2016, end_year: int = 2025) -> list:
    """
    Check which position/year combinations are missing from weather cache.

    Args:
        start_year: First year to check
        end_year: Last year to check

    Returns:
        List of (year, position) tuples that are missing
    """
    missing = []
    positions = ['QB', 'RB', 'WR', 'TE']

    for year in range(start_year, end_year + 1):
        for position in positions:
            cache_file = WEATHER_CACHE_DIR / f"weather_{position.lower()}_{year}.parquet"
            if not cache_file.exists():
                missing.append((year, position))

    return missing
