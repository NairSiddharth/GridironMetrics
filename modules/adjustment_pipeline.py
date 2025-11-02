"""
Adjustment Pipeline Module

Applies sequential adjustment phases to player contribution scores:
- Phase 4: Player-level adjustments (catch rate, blocking, separation, penalties)
- Phase 4.5: Weather-based performance adjustments
- Phase 5: Talent context and sample size dampening

This module centralizes the adjustment logic that was previously scattered
throughout main.py, making it easier to maintain and test.
"""

import polars as pl
from pathlib import Path
from modules.logger import get_logger
from modules.constants import CACHE_DIR
from modules.context_adjustments import ContextAdjustments
from modules.play_by_play import PlayByPlayProcessor

logger = get_logger(__name__)

# Initialize processors (shared with main.py)
context_adj = ContextAdjustments()
pbp_processor = PlayByPlayProcessor()


def calculate_separation_adjustment(nextgen_data: pl.DataFrame, player_name: str, position: str) -> float:
    """
    Calculate separation/cushion adjustment multiplier from NextGen Stats.

    Philosophy:
    - High separation relative to cushion = elite route running → bonus
    - Low separation despite soft cushion = struggles creating space → penalty
    - Separation is more valuable than cushion (cushion is defensive alignment, separation is player skill)

    Multiplier ranges:
    - Elite separation (3.5+ yards): 1.08x
    - Good separation (3.0-3.5 yards): 1.04x
    - Average separation (2.5-3.0 yards): 1.00x (neutral)
    - Below average (<2.5 yards): 0.96x

    Additional context from cushion:
    - Tight coverage (<5 yd cushion) with good separation: extra +0.02x bonus
    - Soft coverage (>7 yd cushion) with poor separation: extra -0.02x penalty

    Args:
        nextgen_data: NextGen Stats DataFrame for the season
        player_name: Player display name
        position: Player position (WR/TE/RB)

    Returns:
        Adjustment multiplier (0.94 - 1.10)
    """
    # Filter to this player's season aggregate
    player_nextgen = nextgen_data.filter(
        pl.col('player_display_name') == player_name
    )

    if len(player_nextgen) == 0:
        return 1.0  # No data, neutral

    # Aggregate across all weeks (weighted by targets)
    total_targets = player_nextgen['targets'].sum()
    if total_targets < 20:  # Minimum threshold for meaningful data
        return 1.0

    # Calculate weighted averages
    avg_separation = (player_nextgen['avg_separation'] * player_nextgen['targets']).sum() / total_targets
    avg_cushion = (player_nextgen['avg_cushion'] * player_nextgen['targets']).sum() / total_targets

    # Handle nulls
    if avg_separation is None or avg_cushion is None:
        return 1.0

    # Base multiplier from separation
    if avg_separation >= 3.5:
        base_mult = 1.08  # Elite
    elif avg_separation >= 3.0:
        base_mult = 1.04  # Good
    elif avg_separation >= 2.5:
        base_mult = 1.00  # Average
    else:
        base_mult = 0.96  # Below average

    # Context adjustment from cushion
    cushion_bonus = 0.0

    # Tight coverage with good separation = elite vs press
    if avg_cushion < 5.0 and avg_separation >= 3.0:
        cushion_bonus = 0.02

    # Soft coverage with poor separation = can't create space even with help
    elif avg_cushion > 7.0 and avg_separation < 2.5:
        cushion_bonus = -0.02

    final_mult = base_mult + cushion_bonus

    # Clamp to reasonable range
    return max(0.94, min(1.10, final_mult))


def apply_phase4_adjustments(contributions: pl.DataFrame, year: int) -> pl.DataFrame:
    """
    Apply Phase 4 player-level adjustments (catch rate, blocking quality, separation metrics, penalties).

    NOTE: Penalties are applied PER-WEEK (only affect the week they occurred).
    Other adjustments are season-wide (catch rate, blocking, separation).

    Adjustments:
    - Catch rate over expected (WR/TE) - season-wide
    - Blocking quality proxy (RB) - season-wide
    - Separation/cushion metrics (WR/TE/RB receiving, 2016+) - season-wide
    - Penalty adjustments (QB/RB/WR/TE skill players) - PER-WEEK

    Args:
        contributions: DataFrame with player weekly contributions
        year: Season year to load PBP data from

    Returns:
        DataFrame with Phase 4 adjustments applied to player_overall_contribution
    """
    logger.info(f"Applying Phase 4 adjustments (catch rate + blocking + separation + penalties) for {year}")

    # Load full season PBP data for catch rate and blocking quality calculations
    try:
        pbp_data = pbp_processor.load_pbp_data(year)
        if pbp_data is None or pbp_data.height == 0:
            logger.warning(f"No PBP data available for {year}, skipping Phase 4 adjustments")
            return contributions
    except Exception as e:
        logger.error(f"Error loading PBP data for Phase 4 adjustments: {str(e)}")
        return contributions

    # Load NextGen Stats for separation/cushion (2016+)
    from modules.nextgen_cache_builder import load_nextgen_cache, NEXTGEN_START_YEAR
    nextgen_data = None
    if year >= NEXTGEN_START_YEAR:
        nextgen_data = load_nextgen_cache(year)
        if nextgen_data is not None:
            logger.info(f"NextGen Stats loaded for {year} ({len(nextgen_data)} records)")

    # Calculate season-wide adjustments for each unique player (catch rate, blocking, separation)
    season_adjustments = []
    unique_players = contributions.select(['player_id', 'player_name', 'team', 'position']).unique()

    for player_row in unique_players.iter_rows(named=True):
        player_id = player_row['player_id']
        player_name = player_row['player_name']
        player_team = player_row['team']
        position = player_row['position']

        catch_rate_mult = 1.0
        blocking_mult = 1.0
        separation_mult = 1.0

        # Catch rate adjustment (WR/TE only)
        if position in ['WR', 'TE']:
            try:
                catch_rate_mult = context_adj.calculate_catch_rate_adjustment(pbp_data, player_id)
            except Exception as e:
                logger.debug(f"Error calculating catch rate for {player_name}: {str(e)}")
                catch_rate_mult = 1.0

        # Blocking quality proxy (RB only)
        if position == 'RB':
            try:
                blocking_mult = context_adj.calculate_blocking_quality_proxy(pbp_data, player_id, player_team)
            except Exception as e:
                logger.debug(f"Error calculating blocking quality for {player_name}: {str(e)}")
                blocking_mult = 1.0

        # Separation/cushion adjustment (WR/TE/RB receiving work, 2016+)
        if position in ['WR', 'TE', 'RB'] and nextgen_data is not None:
            try:
                separation_mult = calculate_separation_adjustment(nextgen_data, player_name, position)
            except Exception as e:
                logger.debug(f"Error calculating separation for {player_name}: {str(e)}")
                separation_mult = 1.0

        season_adjustments.append({
            'player_id': player_id,
            'player_name': player_name,
            'team': player_team,
            'catch_rate_adjustment': catch_rate_mult,
            'blocking_adjustment': blocking_mult,
            'separation_adjustment': separation_mult
        })

    # Join season-wide adjustments
    season_adj_df = pl.DataFrame(season_adjustments)
    contributions = contributions.join(season_adj_df, on=['player_id', 'player_name', 'team'], how='left')

    # Fill missing adjustments with 1.0 (neutral)
    contributions = contributions.with_columns([
        pl.col('catch_rate_adjustment').fill_null(1.0),
        pl.col('blocking_adjustment').fill_null(1.0),
        pl.col('separation_adjustment').fill_null(1.0)
    ])

    # Calculate per-week penalty adjustments (vectorized for performance)
    from modules.penalty_cache_builder import calculate_penalty_adjustments_batch

    # Filter to skill positions for penalty calculation
    skill_positions = contributions.filter(pl.col('position').is_in(['QB', 'RB', 'WR', 'TE']))

    if skill_positions.height > 0:
        # Calculate penalties in batch (vectorized)
        penalty_adj_df = calculate_penalty_adjustments_batch(skill_positions, year)

        # Join back to all contributions
        contributions = contributions.join(
            penalty_adj_df,
            on=['player_id', 'player_name', 'team', 'week'],
            how='left'
        )
    else:
        # No skill position players (shouldn't happen, but handle gracefully)
        contributions = contributions.with_columns([
            pl.lit(1.0).alias('penalty_adjustment')
        ])

    # Fill missing penalty adjustments with 1.0 (neutral)
    contributions = contributions.with_columns([
        pl.col('penalty_adjustment').fill_null(1.0)
    ])

    # Apply combined Phase 4 multiplier to player_overall_contribution
    contributions = contributions.with_columns([
        (pl.col('player_overall_contribution') *
         pl.col('catch_rate_adjustment') *
         pl.col('blocking_adjustment') *
         pl.col('separation_adjustment') *
         pl.col('penalty_adjustment')).alias('player_overall_contribution')
    ])

    logger.info(f"Phase 4 adjustments applied to {len(unique_players)} players")

    return contributions


def apply_phase4_5_weather_adjustments(contributions: pl.DataFrame, year: int, position: str) -> pl.DataFrame:
    """
    Apply Phase 4.5 weather-based performance adjustments.

    Calculates weather adjustments based on player historical performance in
    different weather conditions (temperature, wind, precipitation, environment)
    relative to position averages.

    Args:
        contributions: DataFrame with player weekly contributions
        year: Season year
        position: Position code ('QB', 'RB', 'WR', 'TE')

    Returns:
        DataFrame with weather adjustments applied to player_overall_contribution
    """
    logger.info(f"Applying Phase 4.5 weather adjustments for {position} {year}")

    try:
        # Import weather adjustment function
        from modules.weather_cache_builder import (
            build_weather_performance_cache,
            calculate_weather_adjustment
        )

        # Pre-build cache for this season/position (idempotent - uses cache if exists)
        build_weather_performance_cache(year, position)

        # Load PBP data to get game-level weather info
        pbp_cache_file = Path(CACHE_DIR) / "pbp" / f"pbp_{year}.parquet"
        if not pbp_cache_file.exists():
            logger.warning(f"PBP data not cached for {year}. Skipping weather adjustments.")
            return contributions

        pbp_data = pl.read_parquet(pbp_cache_file)

        # Get unique game weather conditions
        game_weather = pbp_data.select([
            'game_id', 'temp', 'wind', 'weather', 'roof'
        ]).unique(subset=['game_id'])

        # Build list of weather adjustments for each player-week combination
        weather_adjustments = []
        skip_no_game = 0
        skip_no_weather_data = 0
        skip_null_weather = 0

        for row in contributions.iter_rows(named=True):
            player_id = row['player_id']
            week = row['week']
            team = row['team']

            # Normalize team name to uppercase for matching with PBP data
            team_upper = team.upper() if team else None

            # Find the game for this player-week-team combination
            player_game = pbp_data.filter(
                (pl.col('week') == week) &
                ((pl.col('home_team') == team_upper) | (pl.col('away_team') == team_upper))
            ).select('game_id').unique()

            if len(player_game) == 0:
                skip_no_game += 1
                continue

            game_id = player_game['game_id'][0]

            # Get weather for this game
            game_weather_row = game_weather.filter(pl.col('game_id') == game_id)

            if len(game_weather_row) == 0:
                skip_no_weather_data += 1
                continue

            game_temp = game_weather_row['temp'][0]
            game_wind = game_weather_row['wind'][0]
            game_weather_desc = game_weather_row['weather'][0]
            game_roof = game_weather_row['roof'][0]

            # Skip if missing weather data
            if game_temp is None or game_wind is None:
                skip_null_weather += 1
                continue

            # Calculate weather adjustment for this player-game
            weather_adj = calculate_weather_adjustment(
                player_id=player_id,
                season=year,
                position=position,
                game_temp=game_temp,
                game_wind=game_wind,
                game_weather=game_weather_desc,
                game_roof=game_roof
            )

            # Store the adjustment for this player-week-team combination
            weather_adjustments.append({
                'player_id': player_id,
                'week': week,
                'team': team,
                'weather_adjustment': weather_adj
            })

        logger.info(f"Weather adjustment debug for {position}: {len(weather_adjustments)} adjustments calculated, "
                   f"skipped {skip_no_game} (no game), {skip_no_weather_data} (no weather row), {skip_null_weather} (null weather)")

        # Join weather adjustments back to contributions DataFrame
        if len(weather_adjustments) > 0:
            weather_adj_df = pl.DataFrame(weather_adjustments)
            # Drop the weather_adjustment column if it exists (from previous phases)
            if 'weather_adjustment' in contributions.columns:
                contributions = contributions.drop('weather_adjustment')
            # Join the calculated adjustments
            contributions = contributions.join(
                weather_adj_df,
                on=['player_id', 'week', 'team'],
                how='left'
            ).with_columns([
                # Use calculated adjustment if available, otherwise default to 1.0
                pl.col('weather_adjustment').fill_null(1.0)
            ])
        else:
            # No weather adjustments calculated, add default column
            if 'weather_adjustment' not in contributions.columns:
                contributions = contributions.with_columns([
                    pl.lit(1.0).alias('weather_adjustment')
                ])

        # Apply weather adjustment to player_overall_contribution
        contributions = contributions.with_columns([
            (pl.col('player_overall_contribution') *
             pl.col('weather_adjustment')).alias('player_overall_contribution')
        ])

        # Count how many players got adjustments != 1.0
        adjusted_players = contributions.filter(
            pl.col('weather_adjustment') != 1.0
        )['player_id'].n_unique()

        logger.info(f"Phase 4.5 weather adjustments applied to {adjusted_players} {position}s")

        return contributions

    except Exception as e:
        logger.error(f"Error applying weather adjustments: {e}")
        import traceback
        traceback.print_exc()
        return contributions


def apply_phase5_adjustments(contributions: pl.DataFrame, year: int = None) -> pl.DataFrame:
    """
    Apply Phase 5 adjustments (talent context + sample size dampening).

    Two-pass system:
    1. Use baseline scores from contributions to calculate teammate quality
    2. Apply talent adjustment multiplier
    3. Apply sample size dampening based on injury-adjusted games played

    Args:
        contributions: DataFrame with player weekly contributions including baseline scores
        year: Season year for injury adjustment (optional)

    Returns:
        DataFrame with Phase 5 adjustments applied
    """
    logger.info("Applying Phase 5 adjustments (talent context + sample size dampening)")

    # Import injury adjustment functions
    from modules.injury_cache_builder import calculate_injury_adjusted_games

    # Use year parameter for current season
    current_season = year

    # Aggregate to player level with baseline scores and games played
    player_aggregates = contributions.group_by(['player_id', 'player_name', 'team', 'position']).agg([
        pl.col('player_overall_contribution').mean().alias('baseline_score'),
        pl.col('week').count().alias('games_played')
    ])

    # Calculate talent adjustments for each player
    talent_adjustments = []
    for player_row in player_aggregates.iter_rows(named=True):
        player_id = player_row['player_id']
        player_name = player_row['player_name']
        player_team = player_row['team']
        position = player_row['position']

        try:
            talent_mult = context_adj.calculate_teammate_quality_index(
                player_id, player_name, player_team, position, player_aggregates
            )
        except Exception as e:
            logger.debug(f"Error calculating talent adjustment for {player_name}: {str(e)}")
            talent_mult = 1.0

        talent_adjustments.append({
            'player_id': player_id,
            'player_name': player_name,
            'team': player_team,
            'talent_adjustment': talent_mult
        })

    # Join talent adjustments
    talent_df = pl.DataFrame(talent_adjustments)
    contributions = contributions.join(talent_df, on=['player_id', 'player_name', 'team'], how='left')
    contributions = contributions.with_columns(pl.col('talent_adjustment').fill_null(1.0))

    # Apply talent adjustment
    contributions = contributions.with_columns([
        (pl.col('player_overall_contribution') * pl.col('talent_adjustment')).alias('player_overall_contribution')
    ])

    # Apply sample size dampening at player level
    # Group again to get updated scores after talent adjustment
    player_final = contributions.group_by(['player_id', 'player_name', 'team', 'position']).agg([
        pl.col('player_overall_contribution').mean().alias('score_before_dampening'),
        pl.col('week').count().alias('games_played')
    ])

    # Calculate injury-adjusted effective games and dampened scores
    logger.info("Calculating injury-adjusted sample size dampening...")
    dampened_scores = []

    for row in player_final.iter_rows(named=True):
        score = row['score_before_dampening']
        games = row['games_played']
        player_id = row['player_id']  # This IS the GSIS ID
        player_name = row['player_name']

        # Try to calculate injury-adjusted games using player_id (GSIS ID)
        effective_games = games  # Default to actual games

        if current_season is not None and player_id:
            try:
                effective_games = calculate_injury_adjusted_games(
                    player_id, current_season, games, max_games=17
                )
                if effective_games != games:
                    logger.debug(
                        f"{player_name}: {games} actual → {effective_games:.1f} effective games"
                    )
            except Exception as e:
                logger.debug(f"Could not calculate injury adjustment for {player_name}: {e}")

        # Apply dampening using effective games
        dampened = context_adj.apply_sample_size_dampening(score, int(effective_games), full_season_games=17)

        dampened_scores.append({
            'player_id': row['player_id'],
            'player_name': row['player_name'],
            'team': row['team'],
            'dampened_score': dampened
        })

    # Join dampened scores back
    dampened_df = pl.DataFrame(dampened_scores)
    contributions = contributions.join(dampened_df, on=['player_id', 'player_name', 'team'], how='left')

    # Replace player_overall_contribution with dampened score
    contributions = contributions.with_columns([
        pl.col('dampened_score').alias('player_overall_contribution')
    ])

    logger.info(f"Phase 5 adjustments applied")

    return contributions
