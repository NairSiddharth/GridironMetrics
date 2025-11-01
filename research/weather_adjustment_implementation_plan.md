# Weather-Based Performance Adjustment Implementation Plan

## Overview

Implement weather-based performance adjustments for skill players (QB, RB, WR, TE) to reward/penalize players based on their historical performance in different weather conditions relative to position averages. Players who excel in adverse conditions get boosted, while those who struggle get penalized.

**Integration Point:** New Phase 4.5 (Weather Adjustments) - after existing Phase 4 context adjustments (FTN/NextGen/Penalties), before Phase 5 (sample size)

**Target Positions:** QB, RB, WR, TE (all skill positions)

**Data Source:** nflreadpy play-by-play data (already cached) - includes `weather`, `temp`, `wind`, `roof`, `surface` columns

---

## Key Design Principles

### Independent Weather Factors

Rather than combining weather dimensions into complex intersections (Temperature × Wind × Precipitation × Environment would create too many buckets), we evaluate each factor **independently** and combine their effects multiplicatively:

1. **Temperature Effect** - Cold (<32°F), Cool (32-55°F), Moderate (55-75°F), Hot (>75°F)
2. **Wind Effect** - Calm (<10 mph), Moderate (10-20 mph), High (>20 mph)
3. **Precipitation Effect** - Clear, Rain/Snow (extracted from weather description)
4. **Environment Effect** - Dome vs. Outdoor

### Calculation Method

For each weather factor, calculate:
```
player_performance_metric / position_avg_performance_metric = adjustment_factor
```

Apply caps at **0.95-1.05 per factor**, then apply **final cap of 0.90-1.10** on combined effect.

### Example Calculation

**Patrick Mahomes in cold (28°F) + high wind (22 mph) game:**
- Cold weather: His 7.8 YPA vs QB avg 7.2 YPA in cold = 1.08 → capped to 1.05
- High wind: His 7.1 YPA vs QB avg 6.8 YPA in wind = 1.04
- Combined: 1.05 × 1.04 = 1.09 (9% boost)
- Final: clip(1.09, 0.90, 1.10) = 1.09 ✓

**Dome QB (historically struggles outdoors) in cold + wind:**
- Cold weather: His 6.9 YPA vs QB avg 7.2 YPA in cold = 0.96
- High wind: His 6.3 YPA vs QB avg 6.8 YPA in wind = 0.93 → capped to 0.95
- Combined: 0.96 × 0.95 = 0.91 (9% penalty)
- Final: clip(0.91, 0.90, 1.10) = 0.91 ✓

---

## Architecture Overview

### Integration Point: New Phase 4.5

**Current Flow:**
```
Phase 1: Base Contributions
Phase 2: Difficulty Adjustments
Phase 3: Target Share Adjustments
Phase 4: Context Adjustments (FTN, NextGen, Penalties)
Phase 5: Sample Size Dampening
```

**New Flow:**
```
Phase 1: Base Contributions
Phase 2: Difficulty Adjustments
Phase 3: Target Share Adjustments
Phase 4: Context Adjustments (FTN, NextGen, Penalties)
Phase 4.5: Weather Adjustments (NEW)
Phase 5: Sample Size Dampening
```

### File Structure

```
modules/
├── weather_cache_builder.py     [NEW] - Weather analysis and cache builder
└── constants.py                 [MODIFY] - Add weather-related constants

cache/
└── weather/                     [NEW] - Cached weather performance data
    ├── weather_qb_2016.parquet
    ├── weather_rb_2016.parquet
    ├── weather_wr_2016.parquet
    ├── weather_te_2016.parquet
    ├── weather_qb_2017.parquet
    └── ... (through 2025, one file per position per year)

main.py                          [MODIFY] - Integrate into new Phase 4.5
```

---

## Phase 1: Data Module & Cache Builder

### File: `modules/weather_cache_builder.py`

Following established patterns from `penalty_cache_builder.py` and `injury_cache_builder.py`.

```python
"""
Weather Cache Builder Module

Analyzes player performance across different weather conditions and builds
cached weather adjustment factors for Phase 4.5.

Uses play-by-play data (already cached) to extract weather info and calculate
position-level performance baselines and player-specific adjustments.
"""

import polars as pl
import nflreadpy as nfl
from pathlib import Path
from modules.logger import logger
from modules.constants import (
    # Temperature thresholds
    WEATHER_TEMP_COLD,
    WEATHER_TEMP_COOL,
    WEATHER_TEMP_MODERATE,
    WEATHER_TEMP_HOT,

    # Wind thresholds
    WEATHER_WIND_CALM,
    WEATHER_WIND_MODERATE,
    WEATHER_WIND_HIGH,

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
                pl.col('passing_yards').alias('yards'),
                pl.col('pass_touchdown').cast(pl.Int32).alias('td'),
                pl.col('complete_pass').cast(pl.Int32).alias('success')
            ])

        elif position == 'RB':
            # RB rushing plays
            relevant_plays = pbp.filter(
                (pl.col('play_type') == 'run') &
                pl.col('rusher_player_id').is_not_null()
            ).with_columns([
                pl.col('rusher_player_id').alias('player_id'),
                pl.col('rusher_player_name').alias('player_name'),
                pl.col('rushing_yards').alias('yards'),
                pl.col('rush_touchdown').cast(pl.Int32).alias('td'),
                (pl.col('rushing_yards') > 0).cast(pl.Int32).alias('success')
            ])

        elif position in ['WR', 'TE']:
            # WR/TE receiving plays
            relevant_plays = pbp.filter(
                (pl.col('play_type') == 'pass') &
                pl.col('receiver_player_id').is_not_null()
            ).with_columns([
                pl.col('receiver_player_id').alias('player_id'),
                pl.col('receiver_player_name').alias('player_name'),
                pl.col('receiving_yards').alias('yards'),
                pl.col('pass_touchdown').cast(pl.Int32).alias('td'),
                pl.col('complete_pass').cast(pl.Int32).alias('success')
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
        precip_avg = relevant_plays.filter(pl.col('has_precip') == True)['yards'].mean() if len(relevant_plays.filter(pl.col('has_precip') == True)) > 0 else None
        clear_avg = relevant_plays.filter(pl.col('has_precip') == False)['yards'].mean() if len(relevant_plays.filter(pl.col('has_precip') == False)) > 0 else None

        # Environment averages
        dome_avg = relevant_plays.filter(pl.col('env_cat') == 'dome')['yards'].mean() if len(relevant_plays.filter(pl.col('env_cat') == 'dome')) > 0 else None
        outdoor_avg = relevant_plays.filter(pl.col('env_cat') == 'outdoor')['yards'].mean() if len(relevant_plays.filter(pl.col('env_cat') == 'outdoor')) > 0 else None

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

                if len(temp_player_plays) >= WEATHER_MIN_PLAYS and temp_avgs[temp_cat] is not None:
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

                if len(wind_player_plays) >= WEATHER_MIN_PLAYS and wind_avgs[wind_cat] is not None:
                    player_avg = wind_player_plays['yards'].mean()
                    adj = player_avg / wind_avgs[wind_cat]
                    adj = max(WEATHER_FACTOR_MIN, min(WEATHER_FACTOR_MAX, adj))
                else:
                    adj = 1.0

                player_stats[f'wind_{wind_cat}_adj'] = adj

            # Precipitation adjustment
            precip_player_plays = player_plays.filter(pl.col('has_precip') == True)
            if len(precip_player_plays) >= WEATHER_MIN_PLAYS and precip_avg is not None:
                player_avg = precip_player_plays['yards'].mean()
                adj = player_avg / precip_avg
                adj = max(WEATHER_FACTOR_MIN, min(WEATHER_FACTOR_MAX, adj))
            else:
                adj = 1.0
            player_stats['precip_adj'] = adj

            # Environment adjustments
            for env_cat, env_avg in [('dome', dome_avg), ('outdoor', outdoor_avg)]:
                env_player_plays = player_plays.filter(pl.col('env_cat') == env_cat)

                if len(env_player_plays) >= WEATHER_MIN_PLAYS and env_avg is not None:
                    player_avg = env_player_plays['yards'].mean()
                    adj = player_avg / env_avg
                    adj = max(WEATHER_FACTOR_MIN, min(WEATHER_FACTOR_MAX, adj))
                else:
                    adj = 1.0

                player_stats[f'{env_cat}_adj'] = adj

            player_weather_stats.append(player_stats)

        # Convert to DataFrame
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

        player_row = player_weather[0]

        # Categorize current game conditions
        temp_cat = categorize_temperature(game_temp)
        wind_cat = categorize_wind(game_wind)
        has_precip = has_precipitation(game_weather)
        env_cat = categorize_environment(game_roof)

        # Get adjustment factors
        temp_adj = player_row[f'temp_{temp_cat}_adj']
        wind_adj = player_row[f'wind_{wind_cat}_adj']
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
```

---

## Phase 2: Add Constants

### File: `modules/constants.py`

Add weather adjustment constants following existing patterns.

```python
# ============================================================================
# WEATHER ADJUSTMENT CONSTANTS (Phase 4.5)
# ============================================================================

# Temperature Thresholds (Fahrenheit)
WEATHER_TEMP_COLD = 32.0      # Below 32°F = cold
WEATHER_TEMP_COOL = 55.0      # 32-55°F = cool
WEATHER_TEMP_MODERATE = 75.0  # 55-75°F = moderate
# Above 75°F = hot

# Wind Thresholds (mph)
WEATHER_WIND_CALM = 10.0      # Below 10 mph = calm
WEATHER_WIND_MODERATE = 20.0  # 10-20 mph = moderate
# Above 20 mph = high

# Adjustment Factor Caps
WEATHER_FACTOR_MIN = 0.95     # Min adjustment per individual factor
WEATHER_FACTOR_MAX = 1.05     # Max adjustment per individual factor
WEATHER_TOTAL_MIN = 0.90      # Min total weather adjustment
WEATHER_TOTAL_MAX = 1.10      # Max total weather adjustment

# Sample Size Requirements
WEATHER_MIN_PLAYS = 30        # Minimum plays in a condition to calculate adjustment

# Precipitation Keywords (for weather description parsing)
WEATHER_PRECIP_KEYWORDS = [
    'rain', 'snow', 'sleet', 'hail', 'drizzle',
    'showers', 'flurries', 'precipitation'
]
```

---

## Phase 3: Integrate into Phase 4.5

### File: `main.py`

Add new Phase 4.5 after existing Phase 4 in each position's ranking function.

**Integration Points:**
1. `generate_qb_rankings()` - after line ~1200 (after Phase 4)
2. `generate_rb_rankings()` - after line ~1350 (after Phase 4)
3. `generate_wr_rankings()` - after line ~1600 (after Phase 4)
4. `generate_te_rankings()` - after line ~1830 (after Phase 4)

**Example Integration (for QB rankings):**

```python
    # Phase 4: Context-Based Adjustments (FTN Charting + NextGen + Penalties)
    logger.info("Phase 4: Applying context-based adjustments...")

    # ... existing Phase 4 logic ...

    # Phase 4.5: Weather-Based Adjustments
    logger.info("Phase 4.5: Applying weather-based adjustments...")

    from modules.weather_cache_builder import build_weather_performance_cache

    # Pre-build cache for this season (idempotent - uses cache if exists)
    build_weather_performance_cache(year, 'QB')

    weather_adjusted_contributions = {}

    # Load PBP data to get game-level weather
    pbp_data = load_pbp_data(year)
    game_weather = pbp_data.select([
        'game_id', 'temp', 'wind', 'weather', 'roof'
    ]).unique(subset=['game_id'])

    for player_name, contribution in phase4_contributions.items():
        # Get player's GSIS ID
        player_id = get_player_gsis_id(player_name, None, 'QB', year)

        if not player_id:
            weather_adjusted_contributions[player_name] = contribution
            continue

        # Get player's games this season
        player_games = pbp_data.filter(
            pl.col('passer_player_id') == player_id
        ).select('game_id').unique()

        # Calculate average weather adjustment across all games played
        total_weather_adj = 0.0
        game_count = 0

        for game_id in player_games['game_id']:
            game_weather_row = game_weather.filter(pl.col('game_id') == game_id)

            if len(game_weather_row) == 0:
                continue

            game_temp = game_weather_row['temp'][0]
            game_wind = game_weather_row['wind'][0]
            game_weather_desc = game_weather_row['weather'][0]
            game_roof = game_weather_row['roof'][0]

            # Skip if missing weather data
            if game_temp is None or game_wind is None:
                continue

            from modules.weather_cache_builder import calculate_weather_adjustment

            weather_adj = calculate_weather_adjustment(
                player_id=player_id,
                season=year,
                position='QB',
                game_temp=game_temp,
                game_wind=game_wind,
                game_weather=game_weather_desc,
                game_roof=game_roof
            )

            total_weather_adj += weather_adj
            game_count += 1

        # Average weather adjustment across all games
        if game_count > 0:
            avg_weather_adj = total_weather_adj / game_count
        else:
            avg_weather_adj = 1.0

        if avg_weather_adj != 1.0:
            logger.debug(
                f"{player_name}: Weather adjustment = {avg_weather_adj:.3f} "
                f"(avg across {game_count} games)"
            )

        weather_adjusted_contributions[player_name] = contribution * avg_weather_adj
```

---

## Phase 4: Testing Strategy

### 4.1 Unit Tests

Create `test_weather_adjustment.py`:

```python
"""
Test weather adjustment calculations with known scenarios.
"""

from modules.weather_cache_builder import (
    categorize_temperature,
    categorize_wind,
    has_precipitation,
    categorize_environment,
    build_weather_performance_cache,
    calculate_weather_adjustment
)


def test_temperature_categorization():
    """Test temperature buckets."""
    assert categorize_temperature(25.0) == 'cold'
    assert categorize_temperature(45.0) == 'cool'
    assert categorize_temperature(65.0) == 'moderate'
    assert categorize_temperature(85.0) == 'hot'
    print("✓ Temperature categorization works")


def test_wind_categorization():
    """Test wind buckets."""
    assert categorize_wind(5.0) == 'calm'
    assert categorize_wind(15.0) == 'moderate'
    assert categorize_wind(25.0) == 'high'
    print("✓ Wind categorization works")


def test_precipitation_detection():
    """Test precipitation detection."""
    assert has_precipitation("Light Snow") == True
    assert has_precipitation("Rain") == True
    assert has_precipitation("Sunny") == False
    assert has_precipitation("Clear") == False
    print("✓ Precipitation detection works")


def test_environment_categorization():
    """Test environment buckets."""
    assert categorize_environment('dome') == 'dome'
    assert categorize_environment('closed') == 'dome'
    assert categorize_environment('outdoors') == 'outdoor'
    assert categorize_environment('open') == 'outdoor'
    print("✓ Environment categorization works")


def test_cache_building():
    """Test that cache building works for QB 2023."""
    df = build_weather_performance_cache(2023, 'QB')

    assert len(df) > 0, "Should have QB weather data"
    assert 'player_id' in df.columns
    assert 'temp_cold_adj' in df.columns
    assert 'wind_high_adj' in df.columns

    print(f"✓ Built weather cache with {len(df)} QBs for 2023")


def test_mahomes_cold_weather():
    """
    Test Patrick Mahomes in cold weather.
    Mahomes historically performs well in cold weather games.
    """
    # Build cache first
    df = build_weather_performance_cache(2023, 'QB')

    # Find Mahomes
    mahomes = df.filter(pl.col('player_name').str.contains('Mahomes'))

    if len(mahomes) > 0:
        cold_adj = mahomes['temp_cold_adj'][0]
        print(f"\nMahomes cold weather adjustment: {cold_adj:.3f}")

        # Mahomes should perform well in cold (>= 1.0)
        # This is just informational, actual values may vary
        print(f"✓ Mahomes cold weather data loaded")
    else:
        print("✗ Mahomes not found in 2023 data")


def test_weather_adjustment_calculation():
    """
    Test full weather adjustment calculation.
    """
    # Get a real QB from 2023
    df = build_weather_performance_cache(2023, 'QB')

    if len(df) > 0:
        first_qb = df[0]
        player_id = first_qb['player_id']
        player_name = first_qb['player_name']

        # Simulate cold, windy game
        adjustment = calculate_weather_adjustment(
            player_id=player_id,
            season=2023,
            position='QB',
            game_temp=28.0,  # Cold
            game_wind=22.0,  # High wind
            game_weather='Clear',
            game_roof='outdoors'
        )

        print(f"\n{player_name} in cold (28°F) + high wind (22 mph):")
        print(f"  Weather adjustment: {adjustment:.3f}")

        # Should be between 0.90 and 1.10
        assert 0.90 <= adjustment <= 1.10, "Should be within bounds"

        print("✓ Weather adjustment calculation works")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Weather Adjustment System")
    print("=" * 60)

    test_temperature_categorization()
    test_wind_categorization()
    test_precipitation_detection()
    test_environment_categorization()
    test_cache_building()
    test_mahomes_cold_weather()
    test_weather_adjustment_calculation()

    print("\n" + "=" * 60)
    print("All weather adjustment tests passed!")
    print("=" * 60)
```

### 4.2 Integration Test

Run full 2023 pipeline and validate:

```bash
# Run main.py for 2023 with weather adjustments
.venv/Scripts/python.exe main.py

# Check for weather adjustment log messages:
#   "P.Mahomes: Weather adjustment = 1.04 (avg across 17 games)"
#   "J.Allen: Weather adjustment = 1.03 (avg across 17 games)"
#   "Dome QBs: Weather adjustment = 0.96-0.98"
```

### 4.3 Manual Validation

Check specific players in 2023 QB rankings:

**Expected Results:**
- **Patrick Mahomes** (cold-weather specialist): ~1.02-1.05 boost
- **Josh Allen** (cold-weather specialist): ~1.02-1.05 boost
- **Derek Carr** (dome QB moving outdoors): ~0.96-0.98 penalty
- **Matthew Stafford** (dome experience): ~1.00 (neutral)

---

## Phase 5: Pre-Cache Historical Data

Once implementation is complete and tested, pre-cache all years:

```python
from modules.weather_cache_builder import build_all_weather_caches

# Pre-cache 2016-2024 (all years with PBP data)
build_all_weather_caches(2016, 2024)

# Verify cache
from modules.weather_cache_builder import cache_is_up_to_date
missing = cache_is_up_to_date(2016, 2024)
print(f"Missing weather cache files: {missing}")
```

---

## Cache Structure

### Directory: `cache/weather/`

Following established pattern from `cache/pbp/`, `cache/ftn/`, `cache/nextgen/`.

**File Format:** Parquet (following pattern for large datasets)

**Naming Convention:** `weather_{position}_{year}.parquet`

**Example Files:**
```
cache/weather/
├── weather_qb_2016.parquet
├── weather_rb_2016.parquet
├── weather_wr_2016.parquet
├── weather_te_2016.parquet
├── weather_qb_2017.parquet
├── weather_rb_2017.parquet
├── ...
├── weather_qb_2024.parquet
├── weather_rb_2024.parquet
├── weather_wr_2024.parquet
└── weather_te_2024.parquet
```

**Parquet Schema:**
```
player_id: str
player_name: str
temp_cold_adj: float
temp_cool_adj: float
temp_moderate_adj: float
temp_hot_adj: float
wind_calm_adj: float
wind_moderate_adj: float
wind_high_adj: float
precip_adj: float
dome_adj: float
outdoor_adj: float
total_plays: int
```

**Why Parquet:**
- Consistent with PBP, FTN, NextGen cache format
- Efficient storage for columnar data
- Fast read performance
- ~10-50KB per file (small but efficient)

---

## Implementation Checklist

### Phase 1: Module Creation
- [ ] Create `modules/weather_cache_builder.py`
- [ ] Implement `categorize_temperature()`
- [ ] Implement `categorize_wind()`
- [ ] Implement `has_precipitation()`
- [ ] Implement `categorize_environment()`
- [ ] Implement `build_weather_performance_cache()` (core function)
- [ ] Implement `calculate_weather_adjustment()` (main entry point)
- [ ] Implement `build_all_weather_caches()`
- [ ] Implement `cache_is_up_to_date()`

### Phase 2: Constants
- [ ] Add all weather constants to `modules/constants.py`
- [ ] Add temperature thresholds
- [ ] Add wind thresholds
- [ ] Add adjustment caps
- [ ] Add precipitation keywords

### Phase 3: Integration
- [ ] Add Phase 4.5 to `generate_qb_rankings()`
- [ ] Add Phase 4.5 to `generate_rb_rankings()`
- [ ] Add Phase 4.5 to `generate_wr_rankings()`
- [ ] Add Phase 4.5 to `generate_te_rankings()`
- [ ] Add logging for weather adjustments

### Phase 4: Testing
- [ ] Create `test_weather_adjustment.py`
- [ ] Test temperature categorization
- [ ] Test wind categorization
- [ ] Test precipitation detection
- [ ] Test cache building
- [ ] Test known scenarios (Mahomes, Allen, etc.)
- [ ] Run full 2023 pipeline integration test
- [ ] Manually validate QB rankings

### Phase 5: Pre-Caching
- [ ] Pre-cache 2016-2024 weather data (all positions)
- [ ] Verify all cache files present
- [ ] Validate cache file sizes/content

### Phase 6: Validation
- [ ] Compare 2023 rankings before/after weather implementation
- [ ] Verify cold-weather QBs get boosted
- [ ] Verify dome QBs moving outdoors get penalized
- [ ] Check that multipliers are reasonable
- [ ] Validate logging output is clear and informative

---

## Expected Performance Impact

**Weather Adjustment Frequency:**
- Most players: 0.98-1.02 (minimal variance)
- Cold-weather specialists: 1.03-1.05 boost
- Dome QBs outdoors: 0.96-0.98 penalty
- Extreme cases: 0.90-1.10 caps

**Ranking Impact Examples:**
- **Patrick Mahomes** (cold-weather games): +3-5% in cold games
- **Josh Allen** (Buffalo climate): +3-5% overall
- **Derek Carr** (NO→LV transition): -2-4% outdoors
- **Most players**: ±2% variance

**Performance:**
- Cache building: ~10-15 seconds per position per year
- Cache reading: ~0.1 seconds per position
- Per-game adjustment: ~0.001 seconds
- Total Phase 4.5 overhead: ~5-10 seconds for 200 players
- Overall pipeline impact: <5% increase

---

## Success Criteria

✅ **Module Implementation:**
- `weather_cache_builder.py` created and functional
- All functions implemented with proper error handling
- Follows existing module patterns

✅ **Constants Added:**
- All weather thresholds defined in `constants.py`
- Values match agreed-upon buckets

✅ **Integration Complete:**
- Weather adjustments applied in Phase 4.5 for all positions
- Proper logging of weather impacts
- Uses existing PBP cache (no redundant data loading)

✅ **Testing Passed:**
- Known scenarios validated (Mahomes, Allen, etc.)
- Adjustment ranges reasonable (0.90-1.10)
- Full 2023 pipeline runs successfully
- Rankings look reasonable

✅ **Cache System Working:**
- Cache files created in `cache/weather/`
- Parquet format follows naming convention
- Pre-cached 2016-2024 successfully

✅ **Documentation:**
- Implementation plan comprehensive and detailed
- Future sessions can continue work seamlessly
- Code is well-commented and clear

---

## Notes for Future Sessions

### Key Design Decisions:

1. **Integration Point:** New Phase 4.5 (after context adjustments, before sample size)
   - Weather reflects environmental performance ability
   - Applied per-game, then averaged across season

2. **Independent Factors:** Temperature, Wind, Precipitation, Environment evaluated separately
   - Avoids sparse data from complex intersections
   - Each factor contributes ±5% max
   - Combined via multiplication

3. **Caps:** Individual factors 0.95-1.05, total 0.90-1.10
   - Prevents over-adjustment
   - Ensures meaningful but not overwhelming impact

4. **Sample Size:** Minimum 30 plays in a condition
   - Prevents noise from small samples
   - Falls back to position average if insufficient data

5. **Data Source:** Existing PBP cache
   - No new API calls needed
   - Weather data already available in `temp`, `wind`, `weather`, `roof` columns

### Potential Enhancements (Not in Scope):

- **Surface type adjustments:** Grass vs. turf performance
- **Stadium-specific adjustments:** Known difficult venues
- **Time-of-day effects:** Early vs. late games, night games
- **Altitude adjustments:** Denver thin air
- **Career-long weather history:** Multi-year weather profiles

### Troubleshooting:

**Issue: Many players missing from weather cache**
- Solution: Lower `WEATHER_MIN_PLAYS` threshold (currently 30)
- Solution: Verify PBP cache exists for that year
- Solution: Check that plays are being filtered correctly

**Issue: Adjustments seem too harsh/lenient**
- Solution: Adjust `WEATHER_FACTOR_MIN/MAX` in `constants.py`
- Solution: Verify position averages are calculated correctly
- Solution: Check that caps are being applied

**Issue: Cache files not found**
- Solution: Run `build_all_weather_caches()` for missing years
- Solution: Verify `cache/weather/` directory exists
- Solution: Ensure PBP cache exists first

---

## Timeline Estimate

- **Phase 1 (Module):** 60-90 minutes
- **Phase 2 (Constants):** 10 minutes
- **Phase 3 (Integration):** 45-60 minutes
- **Phase 4 (Testing):** 30-45 minutes
- **Phase 5 (Pre-caching):** 15-20 minutes (build time)
- **Phase 6 (Validation):** 20-30 minutes

**Total Estimated Time:** 3-4 hours

---

## End of Implementation Plan

This plan is comprehensive enough for any session to pick up and continue implementation seamlessly. All design decisions, integration points, testing strategies, and edge cases are documented.
