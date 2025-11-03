"""
Player Props Data Aggregator

Calculates rolling average baselines for player prop projections using
GridironMetrics' established weighted cumulative methodology.

Data Sources:
- cache/positional_player_stats/{position}/{position}-{year}.csv

Methodology:
- Game weighting: 0.25 → 0.50 → 0.75 → 1.0 for first 4 games, 1.0 thereafter
- Formula: weighted_cumulative_sum / cumulative_weight_sum
- Calculated through week N only (no future data leakage)
"""

import polars as pl
from pathlib import Path
from typing import Dict, Optional, List
from modules.logger import get_logger
from modules.constants import CACHE_DIR

logger = get_logger(__name__)


def calculate_game_weights(num_games: int) -> List[float]:
    """
    Calculate game weights following GridironMetrics methodology.

    Game weights ramp over first 4 games to downweight early-season volatility:
    - Game 1: 0.25
    - Game 2: 0.50
    - Game 3: 0.75
    - Game 4+: 1.0

    Args:
        num_games: Total number of games

    Returns:
        List of weights for each game
    """
    weights = []
    for i in range(num_games):
        if i == 0:
            weights.append(0.25)
        elif i == 1:
            weights.append(0.50)
        elif i == 2:
            weights.append(0.75)
        else:
            weights.append(1.0)
    return weights


def calculate_weighted_rolling_average(
    player_stats: pl.DataFrame,
    stat_column: str,
    through_week: int
) -> float:
    """
    Calculate weighted rolling average with RECENCY WEIGHTING through specified week.

    NEW Tier 2 weighting scheme (emphasizes recent performance):
    - Last 3 games from most recent: 1.5x weight
    - Games 4-6 from most recent: 1.0x weight
    - Games 7+ from most recent: 0.75x weight

    This replaced the old early-game dampening system (0.25, 0.50, 0.75, 1.0)
    which was designed for season-long rankings, not weekly projections.

    Formula: weighted_sum / cumulative_weight_sum

    Args:
        player_stats: DataFrame with weekly stats (sorted by week)
        stat_column: Column name (e.g., 'passing_yards', 'rushing_tds')
        through_week: Calculate through this week (no future data)

    Returns:
        Recency-weighted rolling average

    Example:
        For 8 games [292, 315, 245, 267, 289, 301, 278, 264] (oldest to newest):
        Game 1-2 (oldest): 0.75x weight
        Game 3-5: 1.0x weight
        Game 6-8 (most recent): 1.5x weight
        This emphasizes recent form over old performance
    """
    # Filter to through specified week
    filtered = player_stats.filter(pl.col('week') <= through_week).sort('week')

    if len(filtered) == 0:
        return 0.0

    # Get stat values (oldest to newest)
    stat_values = filtered[stat_column].to_list()
    num_games = len(stat_values)

    # Calculate recency-based weights (from oldest to newest)
    weights = []
    for i in range(num_games):
        games_from_end = num_games - i  # How many games back from most recent

        if games_from_end <= 3:
            # Last 3 games: highest weight
            weights.append(1.5)
        elif games_from_end <= 6:
            # Games 4-6 back: normal weight
            weights.append(1.0)
        else:
            # Games 7+ back: reduced weight
            weights.append(0.75)

    # Calculate weighted sum
    weighted_sum = sum(val * weight for val, weight in zip(stat_values, weights))
    cumulative_weight = sum(weights)

    if cumulative_weight == 0:
        return 0.0

    average = weighted_sum / cumulative_weight

    logger.debug(
        f"Weighted rolling average for {stat_column}: "
        f"{num_games} games, recency weights={weights}, avg={average:.2f}"
    )

    return average


def get_simple_average(
    player_stats: pl.DataFrame,
    stat_column: str,
    last_n_games: int,
    through_week: int
) -> float:
    """
    Calculate simple average over last N games.

    Used for L3 and L5 averages in output display (not for projections).

    Args:
        player_stats: DataFrame with weekly stats
        stat_column: Column name
        last_n_games: Number of recent games (3 or 5)
        through_week: Calculate through this week

    Returns:
        Simple average over last N games
    """
    # Filter to through specified week and sort
    filtered = player_stats.filter(pl.col('week') <= through_week).sort('week')

    if len(filtered) == 0:
        return 0.0

    # Take last N games
    recent = filtered.tail(last_n_games)
    stat_values = recent[stat_column].to_list()

    if len(stat_values) == 0:
        return 0.0

    return sum(stat_values) / len(stat_values)


def get_historical_averages(
    player_id: str,
    season: int,
    week: int,
    position: str,
    stat_column: str
) -> Dict[str, float]:
    """
    Get L3, L5, and season averages for display in output tables.

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Current week
        position: Player position (QB, RB, WR, TE)
        stat_column: Stat to calculate (e.g., 'passing_yards')

    Returns:
        {
            'last_3_avg': float,
            'last_5_avg': float,
            'season_avg': float (weighted)
        }
    """
    # Load player stats
    stats_file = Path(CACHE_DIR) / "positional_player_stats" / position.lower() / f"{position.lower()}-{season}.csv"

    if not stats_file.exists():
        logger.warning(f"Stats file not found: {stats_file}")
        return {'last_3_avg': 0.0, 'last_5_avg': 0.0, 'season_avg': 0.0}

    # Load and filter to player
    df = pl.read_csv(stats_file)
    player_stats = df.filter(pl.col('player_id') == player_id)

    if len(player_stats) == 0:
        logger.warning(f"No stats found for player {player_id} in {season}")
        return {'last_3_avg': 0.0, 'last_5_avg': 0.0, 'season_avg': 0.0}

    # Calculate averages
    last_3 = get_simple_average(player_stats, stat_column, 3, week)
    last_5 = get_simple_average(player_stats, stat_column, 5, week)
    season = calculate_weighted_rolling_average(player_stats, stat_column, week)

    return {
        'last_3_avg': last_3,
        'last_5_avg': last_5,
        'season_avg': season
    }


def get_career_averages(
    player_id: str,
    current_season: int,
    position: str,
    stat_columns: List[str],
    lookback_years: int = 3
) -> Dict[str, float]:
    """
    Calculate career per-game averages from previous seasons.

    Args:
        player_id: Player GSIS ID
        current_season: Current season year
        position: Player position
        stat_columns: List of stat columns to calculate
        lookback_years: Number of prior seasons to include (default 3)

    Returns:
        Dict of stat_name -> career_per_game_average
    """
    career_averages = {}
    total_games = 0
    career_totals = {stat: 0.0 for stat in stat_columns}

    # Load previous seasons
    for year_offset in range(1, lookback_years + 1):
        past_season = current_season - year_offset

        stats_file = Path(CACHE_DIR) / "positional_player_stats" / position.lower() / f"{position.lower()}-{past_season}.csv"

        if not stats_file.exists():
            continue

        try:
            df = pl.read_csv(stats_file)
            player_stats = df.filter(pl.col('player_id') == player_id)

            if len(player_stats) == 0:
                continue

            # Accumulate totals
            games_in_season = len(player_stats)
            total_games += games_in_season

            for stat_col in stat_columns:
                if stat_col in player_stats.columns:
                    season_total = player_stats[stat_col].sum()
                    career_totals[stat_col] += season_total

        except Exception as e:
            logger.debug(f"Error loading {past_season} data for career average: {e}")
            continue

    # Calculate per-game averages
    if total_games > 0:
        for stat_col in stat_columns:
            career_averages[stat_col] = career_totals[stat_col] / total_games
    else:
        # No career data available
        for stat_col in stat_columns:
            career_averages[stat_col] = 0.0

    logger.debug(
        f"Career averages for {player_id}: {total_games} games over {lookback_years} years, "
        f"{', '.join(f'{k}={v:.1f}' for k, v in career_averages.items())}"
    )

    return career_averages


def get_player_baseline_projections(
    player_id: str,
    season: int,
    week: int,
    position: str
) -> Dict[str, float]:
    """
    Get baseline projections for all prop types for a player.

    Uses weighted rolling average through specified week, blended with career mean.

    Tier 2 Regression to Mean:
    - 80% current season (recency-weighted rolling average)
    - 20% career average (last 3 seasons per-game average)
    - Prevents overreaction to hot/cold streaks
    - Anchors projections to player's established baseline

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Current week (project for next week using data through this week)
        position: Player position (QB, RB, WR, TE)

    Returns:
        {
            'passing_yards': float,
            'passing_tds': float,
            'passing_interceptions': float,
            'rushing_yards': float,
            'rushing_tds': float,
            'receptions': float,
            'receiving_yards': float,
            'receiving_tds': float,
            'games_played': int
        }
    """
    # Load player stats
    stats_file = Path(CACHE_DIR) / "positional_player_stats" / position.lower() / f"{position.lower()}-{season}.csv"

    if not stats_file.exists():
        logger.warning(f"Stats file not found: {stats_file}")
        return {}

    # Load and filter to player
    df = pl.read_csv(stats_file)
    player_stats = df.filter(pl.col('player_id') == player_id)

    if len(player_stats) == 0:
        logger.warning(f"No stats found for player {player_id} in {season}")
        return {}

    # Filter through specified week
    player_stats_filtered = player_stats.filter(pl.col('week') <= week).sort('week')
    games_played = len(player_stats_filtered)

    # Calculate baseline projections for each stat
    projections = {'games_played': games_played}

    # Define stat columns by position
    stat_columns_by_position = {
        'QB': ['passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards', 'rushing_tds'],
        'RB': ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds'],
        'WR': ['receptions', 'receiving_yards', 'receiving_tds'],
        'TE': ['receptions', 'receiving_yards', 'receiving_tds'],
    }

    stat_columns = stat_columns_by_position.get(position.upper(), [])

    # Get career averages for regression to mean
    career_averages = get_career_averages(
        player_id=player_id,
        current_season=season,
        position=position,
        stat_columns=stat_columns,
        lookback_years=3
    )

    # Calculate current season baseline + blend with career average
    for stat_col in stat_columns:
        if stat_col in player_stats_filtered.columns:
            # Current season baseline (recency-weighted)
            current_season_avg = calculate_weighted_rolling_average(
                player_stats_filtered,
                stat_col,
                week
            )

            # Career average
            career_avg = career_averages.get(stat_col, 0.0)

            # Blend: 80% current season + 20% career average
            if career_avg > 0:
                blended = 0.80 * current_season_avg + 0.20 * career_avg
                projections[stat_col] = blended
                logger.debug(
                    f"{stat_col}: current={current_season_avg:.1f}, career={career_avg:.1f}, "
                    f"blended={blended:.1f}"
                )
            else:
                # No career data - use current season only
                projections[stat_col] = current_season_avg
        else:
            projections[stat_col] = 0.0

    logger.info(
        f"Baseline projections for {player_id} ({position}) through week {week}: "
        f"{games_played} games, "
        f"{', '.join(f'{k}={v:.1f}' for k, v in projections.items() if k != 'games_played')}"
    )

    return projections


def calculate_stat_variance(
    player_stats: pl.DataFrame,
    stat_column: str,
    through_week: int
) -> float:
    """
    Calculate coefficient of variation for confidence scoring.

    CV (Coefficient of Variation) = standard_deviation / mean

    Measures relative variability:
    - Low CV (<0.15): Consistent performance (high confidence)
    - Medium CV (0.15-0.25): Moderate variability
    - High CV (>0.25): High volatility (low confidence)

    Args:
        player_stats: DataFrame with weekly stats
        stat_column: Column name
        through_week: Calculate through this week

    Returns:
        Coefficient of variation (CV)
    """
    # Filter to through specified week
    filtered = player_stats.filter(pl.col('week') <= through_week).sort('week')

    if len(filtered) < 2:
        return 1.0  # High variance for insufficient data

    stat_values = filtered[stat_column].to_list()

    if len(stat_values) == 0:
        return 1.0

    mean_val = sum(stat_values) / len(stat_values)

    if mean_val == 0:
        return 1.0  # Avoid division by zero

    # Calculate standard deviation
    variance = sum((x - mean_val) ** 2 for x in stat_values) / len(stat_values)
    std_dev = variance ** 0.5

    cv = std_dev / mean_val

    logger.debug(f"CV for {stat_column}: mean={mean_val:.2f}, std={std_dev:.2f}, CV={cv:.3f}")

    return cv


def get_player_stat_summary(
    player_id: str,
    season: int,
    week: int,
    position: str,
    stat_column: str
) -> Dict[str, float]:
    """
    Get complete statistical summary for a player stat.

    Combines baseline projection, historical averages, and variance.

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Current week
        position: Player position
        stat_column: Stat column name

    Returns:
        {
            'baseline': float (weighted rolling average),
            'last_3_avg': float,
            'last_5_avg': float,
            'season_avg': float,
            'variance': float (CV),
            'games_played': int
        }
    """
    # Load player stats
    stats_file = Path(CACHE_DIR) / "positional_player_stats" / position.lower() / f"{position.lower()}-{season}.csv"

    if not stats_file.exists():
        return {}

    df = pl.read_csv(stats_file)
    player_stats = df.filter(pl.col('player_id') == player_id).filter(pl.col('week') <= week).sort('week')

    if len(player_stats) == 0:
        return {}

    # Calculate all metrics
    baseline = calculate_weighted_rolling_average(player_stats, stat_column, week)
    last_3 = get_simple_average(player_stats, stat_column, 3, week)
    last_5 = get_simple_average(player_stats, stat_column, 5, week)
    variance = calculate_stat_variance(player_stats, stat_column, week)

    return {
        'baseline': baseline,
        'last_3_avg': last_3,
        'last_5_avg': last_5,
        'season_avg': baseline,  # Same as baseline (weighted)
        'variance': variance,
        'games_played': len(player_stats)
    }


if __name__ == "__main__":
    # Test on Patrick Mahomes 2024 Week 10
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Mahomes GSIS ID (you'll need to look this up from roster)
    # This is just a test structure
    print("=== Testing prop_data_aggregator.py ===")
    print("Module loaded successfully")
    print("\nTo test:")
    print("1. Get player GSIS ID from cache/rosters/rosters-2024.csv")
    print("2. Call get_player_baseline_projections(player_id, 2024, 9, 'QB')")
    print("3. Verify weighted rolling average calculation")
