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


def calculate_success_rate_adjustments_batch(contributions: pl.DataFrame, pbp_data: pl.DataFrame, year: int) -> pl.DataFrame:
    """
    Calculate per-week success rate adjustments for offensive players.

    Philosophy:
    - High success rate players consistently move the chains and maintain drives
    - Success rate is calculated per week per player (minimum 20 plays required)
    - Applied as a multiplier to weekly contribution scores

    Success Rate Tiers:
    - Critical (>70%): 1.15x - Elite efficiency
    - High (60-70%): 1.10x - Above average
    - Average (50-60%): 1.05x - Baseline
    - Low (<50%): 0.92x - Below average

    Args:
        contributions: DataFrame with player weekly contributions
        pbp_data: Play-by-play data for the season
        year: Season year

    Returns:
        DataFrame with columns: player_id, player_name, team, week, success_rate_adjustment
    """
    from modules.constants import (
        SUCCESS_RATE_CRITICAL_MULTIPLIER,
        SUCCESS_RATE_HIGH_MULTIPLIER,
        SUCCESS_RATE_AVERAGE_MULTIPLIER,
        SUCCESS_RATE_LOW_MULTIPLIER,
        SUCCESS_RATE_MIN_PLAYS,
        PHASE_6_SUCCESS_RATE_START_YEAR
    )

    # Check if success rate data is available for this year
    if year < PHASE_6_SUCCESS_RATE_START_YEAR:
        logger.debug(f"Success rate data not available for {year} (requires {PHASE_6_SUCCESS_RATE_START_YEAR}+)")
        return contributions.select(['player_id', 'player_name', 'team', 'week']).with_columns([
            pl.lit(1.0).alias('success_rate_adjustment')
        ])

    # Check if success column exists in PBP data
    if 'success' not in pbp_data.columns:
        logger.warning(f"'success' column not found in PBP data for {year}, skipping success rate adjustments")
        return contributions.select(['player_id', 'player_name', 'team', 'week']).with_columns([
            pl.lit(1.0).alias('success_rate_adjustment')
        ])

    # Filter to offensive plays (pass/run) with non-null success values
    offensive_plays = pbp_data.filter(
        (pl.col('play_type').is_in(['pass', 'run'])) &
        (pl.col('success').is_not_null())
    )

    # Calculate success rate per player per week
    # For QB: use passer_id
    # For skill positions: use receiver_id (pass) or rusher_id (run)

    qb_success = offensive_plays.filter(
        (pl.col('play_type') == 'pass') &
        (pl.col('passer_id').is_not_null())
    ).group_by(['passer_id', 'week']).agg([
        pl.col('success').sum().alias('successful_plays'),
        pl.len().alias('total_plays'),
        (pl.col('success').sum() / pl.len() * 100).alias('success_rate')
    ]).rename({'passer_id': 'player_id'})

    rb_rush_success = offensive_plays.filter(
        (pl.col('play_type') == 'run') &
        (pl.col('rusher_id').is_not_null())
    ).group_by(['rusher_id', 'week']).agg([
        pl.col('success').sum().alias('successful_plays'),
        pl.len().alias('total_plays'),
        (pl.col('success').sum() / pl.len() * 100).alias('success_rate')
    ]).rename({'rusher_id': 'player_id'})

    rec_success = offensive_plays.filter(
        (pl.col('play_type') == 'pass') &
        (pl.col('receiver_id').is_not_null()) &
        (pl.col('complete_pass') == 1)  # Only count completions for receivers
    ).group_by(['receiver_id', 'week']).agg([
        pl.col('success').sum().alias('successful_plays'),
        pl.len().alias('total_plays'),
        (pl.col('success').sum() / pl.len() * 100).alias('success_rate')
    ]).rename({'receiver_id': 'player_id'})

    # Combine all success rates
    all_success = pl.concat([qb_success, rb_rush_success, rec_success])

    # Filter to minimum play threshold and apply multipliers
    all_success = all_success.filter(pl.col('total_plays') >= SUCCESS_RATE_MIN_PLAYS)

    all_success = all_success.with_columns([
        pl.when(pl.col('success_rate') > 70.0)
        .then(pl.lit(SUCCESS_RATE_CRITICAL_MULTIPLIER))
        .when(pl.col('success_rate') > 60.0)
        .then(pl.lit(SUCCESS_RATE_HIGH_MULTIPLIER))
        .when(pl.col('success_rate') > 50.0)
        .then(pl.lit(SUCCESS_RATE_AVERAGE_MULTIPLIER))
        .otherwise(pl.lit(SUCCESS_RATE_LOW_MULTIPLIER))
        .alias('success_rate_adjustment')
    ])

    # Join back to contributions to get player_name and team
    result = contributions.select(['player_id', 'player_name', 'team', 'week']).join(
        all_success.select(['player_id', 'week', 'success_rate_adjustment']),
        on=['player_id', 'week'],
        how='left'
    )

    # Fill missing with 1.0 (neutral) - happens when player doesn't meet minimum play threshold
    result = result.with_columns([
        pl.col('success_rate_adjustment').fill_null(1.0)
    ])

    logger.info(f"Success rate adjustments calculated for {all_success.height} player-weeks")

    return result


def calculate_route_location_adjustments(contributions: pl.DataFrame, pbp_data: pl.DataFrame, year: int, position: str) -> pl.DataFrame:
    """
    Calculate season-wide route location adjustments for receivers (WR/TE).

    Philosophy:
    - Rewards receivers who consistently catch passes in difficult locations
    - Deep middle routes are most valuable (contested, tight windows)
    - Slot bias mitigation: short middle routes get small bonus
    - Season-wide adjustment based on weighted route distribution

    Route Location Multipliers:
    - Deep Middle (15+ yards): 1.12x - Most difficult, contested
    - Intermediate Middle (8-14 yards): 1.06x
    - Short Middle (<8 yards): 1.03x - Slot bonus
    - Deep Sideline (15+ yards): 1.05x
    - Intermediate Sideline (8-14 yards): 1.00x - Baseline
    - Short Sideline (<8 yards): 0.95x - Easiest catches

    Args:
        contributions: DataFrame with player contributions
        pbp_data: Play-by-play data for the season
        year: Season year
        position: Player position (WR or TE)

    Returns:
        DataFrame with columns: player_id, player_name, route_location_adjustment
    """
    from modules.constants import (
        ROUTE_DEEP_MIDDLE_MULTIPLIER,
        ROUTE_INT_MIDDLE_MULTIPLIER,
        ROUTE_SHORT_MIDDLE_MULTIPLIER,
        ROUTE_DEEP_SIDELINE_MULTIPLIER,
        ROUTE_INT_SIDELINE_MULTIPLIER,
        ROUTE_SHORT_SIDELINE_MULTIPLIER,
        ROUTE_LOCATION_MIN_TARGETS,
        PHASE_6_ROUTE_LOCATION_START_YEAR
    )

    # Only apply to receivers (WR/TE)
    if position not in ['WR', 'TE']:
        return contributions.select(['player_id', 'player_name']).unique().with_columns([
            pl.lit(1.0).alias('route_location_adjustment')
        ])

    # Check if route location data is available for this year
    if year < PHASE_6_ROUTE_LOCATION_START_YEAR:
        logger.debug(f"Route location data not available for {year} (requires {PHASE_6_ROUTE_LOCATION_START_YEAR}+)")
        return contributions.select(['player_id', 'player_name']).unique().with_columns([
            pl.lit(1.0).alias('route_location_adjustment')
        ])

    # Check if required columns exist in PBP data
    if 'pass_location' not in pbp_data.columns or 'air_yards' not in pbp_data.columns:
        logger.warning(f"Route location columns not found in PBP data for {year}, skipping route location adjustments")
        return contributions.select(['player_id', 'player_name']).unique().with_columns([
            pl.lit(1.0).alias('route_location_adjustment')
        ])

    # Filter to pass plays with receiver and location data
    receiving_plays = pbp_data.filter(
        (pl.col('play_type') == 'pass') &
        (pl.col('receiver_id').is_not_null()) &
        (pl.col('pass_location').is_not_null()) &
        (pl.col('air_yards').is_not_null())
    )

    # Calculate route location multiplier for each play
    receiving_plays = receiving_plays.with_columns([
        pl.when(
            (pl.col('pass_location') == 'middle') & (pl.col('air_yards') >= 15)
        ).then(pl.lit(ROUTE_DEEP_MIDDLE_MULTIPLIER))
        .when(
            (pl.col('pass_location') == 'middle') & (pl.col('air_yards') >= 8) & (pl.col('air_yards') < 15)
        ).then(pl.lit(ROUTE_INT_MIDDLE_MULTIPLIER))
        .when(
            (pl.col('pass_location') == 'middle') & (pl.col('air_yards') < 8)
        ).then(pl.lit(ROUTE_SHORT_MIDDLE_MULTIPLIER))
        .when(
            (pl.col('pass_location').is_in(['left', 'right'])) & (pl.col('air_yards') >= 15)
        ).then(pl.lit(ROUTE_DEEP_SIDELINE_MULTIPLIER))
        .when(
            (pl.col('pass_location').is_in(['left', 'right'])) & (pl.col('air_yards') >= 8) & (pl.col('air_yards') < 15)
        ).then(pl.lit(ROUTE_INT_SIDELINE_MULTIPLIER))
        .when(
            (pl.col('pass_location').is_in(['left', 'right'])) & (pl.col('air_yards') < 8)
        ).then(pl.lit(ROUTE_SHORT_SIDELINE_MULTIPLIER))
        .otherwise(pl.lit(1.0))
        .alias('route_multiplier')
    ])

    # Calculate weighted average route multiplier per receiver for the season
    route_adjustments = receiving_plays.group_by('receiver_id').agg([
        pl.col('route_multiplier').mean().alias('route_location_adjustment'),
        pl.len().alias('total_targets')
    ]).rename({'receiver_id': 'player_id'})

    # Filter to minimum target threshold
    route_adjustments = route_adjustments.filter(
        pl.col('total_targets') >= ROUTE_LOCATION_MIN_TARGETS
    )

    # Join back to contributions to get player_name
    result = contributions.select(['player_id', 'player_name']).unique().join(
        route_adjustments.select(['player_id', 'route_location_adjustment']),
        on='player_id',
        how='left'
    )

    # Fill missing with 1.0 (neutral) - happens when player doesn't meet minimum target threshold
    result = result.with_columns([
        pl.col('route_location_adjustment').fill_null(1.0)
    ])

    logger.info(f"Route location adjustments calculated for {route_adjustments.height} receivers")

    return result


def calculate_turnover_attribution_penalties_batch(contributions: pl.DataFrame, pbp_data: pl.DataFrame, year: int, position: str) -> pl.DataFrame:
    """
    Calculate per-week turnover attribution penalties for offensive players.

    Philosophy:
    - Turnovers are costly plays that directly harm team success
    - QB interceptions get base penalty, reduced if under pressure (not their fault)
    - WR tipped interceptions get penalty (receiver caused the turnover)
    - RB fumbles get penalties based on whether the fumble was recovered

    Penalties:
    - QB interceptions: -15 base, -10 under pressure (2016+)
    - WR tipped interceptions: -8 (requires parsing)
    - RB fumbles: -5 if recovered, -12 if lost

    Args:
        contributions: DataFrame with player weekly contributions
        pbp_data: Play-by-play data for the season
        year: Season year
        position: Player position (QB/RB/WR/TE)

    Returns:
        DataFrame with columns: player_id, player_name, team, week, turnover_penalty
    """
    from modules.constants import (
        QB_INT_BASE_PENALTY,
        QB_INT_PRESSURE_REDUCTION,
        WR_TIP_INT_PENALTY,
        RB_FUMBLE_RECOVERED_PENALTY,
        RB_FUMBLE_LOST_PENALTY,
        PHASE_6_PRESSURE_DATA_START_YEAR
    )

    # Check if position has turnovers to track
    if position not in ['QB', 'RB', 'WR', 'TE']:
        return contributions.select(['player_id', 'player_name', 'team', 'week']).unique().with_columns([
            pl.lit(0.0).alias('turnover_penalty')
        ])

    penalties = []

    if position == 'QB':
        # QB interceptions
        if 'interception' not in pbp_data.columns or 'passer_id' not in pbp_data.columns:
            logger.debug(f"Interception columns not found in PBP data for {year}, skipping QB turnover penalties")
            return contributions.select(['player_id', 'player_name', 'team', 'week']).unique().with_columns([
                pl.lit(0.0).alias('turnover_penalty')
            ])

        # Check if pressure data is available
        has_pressure = 'qb_hit' in pbp_data.columns and year >= PHASE_6_PRESSURE_DATA_START_YEAR

        # Filter to interceptions
        interceptions = pbp_data.filter(
            (pl.col('interception') == 1) &
            (pl.col('passer_id').is_not_null())
        )

        if interceptions.height > 0:
            # Calculate penalty per interception
            if has_pressure:
                # Apply pressure reduction if QB was hit
                interceptions = interceptions.with_columns([
                    pl.when(pl.col('qb_hit') == 1)
                    .then(pl.lit(QB_INT_BASE_PENALTY * (1 - QB_INT_PRESSURE_REDUCTION)))
                    .otherwise(pl.lit(QB_INT_BASE_PENALTY))
                    .alias('int_penalty')
                ])
            else:
                # No pressure data, apply base penalty
                interceptions = interceptions.with_columns([
                    pl.lit(QB_INT_BASE_PENALTY).alias('int_penalty')
                ])

            # Group by player and week to sum penalties
            qb_penalties = interceptions.group_by(['passer_id', 'posteam', 'week']).agg([
                pl.col('int_penalty').sum().alias('turnover_penalty')
            ]).rename({
                'passer_id': 'player_id',
                'posteam': 'team'
            })

            # Normalize team codes to uppercase to match player stats
            qb_penalties = qb_penalties.with_columns([
                pl.col('team').str.to_uppercase().alias('team')
            ])

            penalties.append(qb_penalties)

    elif position == 'RB':
        # RB fumbles
        if 'fumble' not in pbp_data.columns or 'fumbled_1_player_id' not in pbp_data.columns:
            logger.debug(f"Fumble columns not found in PBP data for {year}, skipping RB turnover penalties")
            return contributions.select(['player_id', 'player_name', 'team', 'week']).unique().with_columns([
                pl.lit(0.0).alias('turnover_penalty')
            ])

        # Filter to fumbles by RBs (rushers)
        fumbles = pbp_data.filter(
            (pl.col('fumble') == 1) &
            (pl.col('fumbled_1_player_id').is_not_null()) &
            (pl.col('play_type') == 'run')  # RB fumbles on rushing plays
        )

        if fumbles.height > 0:
            # Check if fumble was lost (fumble_lost column)
            if 'fumble_lost' in fumbles.columns:
                fumbles = fumbles.with_columns([
                    pl.when(pl.col('fumble_lost') == 1)
                    .then(pl.lit(RB_FUMBLE_LOST_PENALTY))
                    .otherwise(pl.lit(RB_FUMBLE_RECOVERED_PENALTY))
                    .alias('fumble_penalty')
                ])
            else:
                # No fumble_lost data, assume all fumbles are lost (conservative)
                fumbles = fumbles.with_columns([
                    pl.lit(RB_FUMBLE_LOST_PENALTY).alias('fumble_penalty')
                ])

            # Group by player and week to sum penalties
            rb_penalties = fumbles.group_by(['fumbled_1_player_id', 'posteam', 'week']).agg([
                pl.col('fumble_penalty').sum().alias('turnover_penalty')
            ]).rename({
                'fumbled_1_player_id': 'player_id',
                'posteam': 'team'
            })

            # Normalize team codes to uppercase to match player stats
            rb_penalties = rb_penalties.with_columns([
                pl.col('team').str.to_uppercase().alias('team')
            ])

            penalties.append(rb_penalties)

    elif position in ['WR', 'TE']:
        # WR/TE tipped interceptions
        # Note: This requires parsing the play description or using the 'pass_defense_1_player_id' column
        # For now, we'll skip this as it requires more complex parsing
        logger.debug(f"WR/TE tipped interception tracking not yet implemented for {year}")
        return contributions.select(['player_id', 'player_name', 'team', 'week']).unique().with_columns([
            pl.lit(0.0).alias('turnover_penalty')
        ])

    # Combine all penalties
    if len(penalties) == 0:
        # No turnovers found for this position
        return contributions.select(['player_id', 'player_name', 'team', 'week']).unique().with_columns([
            pl.lit(0.0).alias('turnover_penalty')
        ])

    all_penalties = pl.concat(penalties)

    # Join back to contributions to get player_name
    result = contributions.select(['player_id', 'player_name', 'team', 'week']).unique().join(
        all_penalties.select(['player_id', 'team', 'week', 'turnover_penalty']),
        on=['player_id', 'team', 'week'],
        how='left'
    )

    # Fill missing with 0 (no turnovers)
    result = result.with_columns([
        pl.col('turnover_penalty').fill_null(0.0)
    ])

    # Count how many player-weeks have penalties
    penalty_count = result.filter(pl.col('turnover_penalty') < 0).height
    logger.info(f"Turnover penalties calculated for {penalty_count} {position} player-weeks")

    return result


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

    # Calculate per-week success rate adjustments (Phase 6.1)
    success_rate_adj_df = calculate_success_rate_adjustments_batch(contributions, pbp_data, year)

    # Join success rate adjustments
    contributions = contributions.join(
        success_rate_adj_df.select(['player_id', 'player_name', 'team', 'week', 'success_rate_adjustment']),
        on=['player_id', 'player_name', 'team', 'week'],
        how='left'
    )

    # Fill missing success rate adjustments with 1.0 (neutral)
    contributions = contributions.with_columns([
        pl.col('success_rate_adjustment').fill_null(1.0)
    ])

    # Calculate season-wide route location adjustments (Phase 6.3) for WR/TE
    # Process each position separately since route location only applies to receivers
    route_location_adjustments = []

    for position in unique_players['position'].unique():
        position_contribs = contributions.filter(pl.col('position') == position)
        if position_contribs.height > 0:
            route_adj_df = calculate_route_location_adjustments(position_contribs, pbp_data, year, position)
            route_location_adjustments.append(route_adj_df)

    # Combine all route location adjustments
    if route_location_adjustments:
        all_route_adj = pl.concat(route_location_adjustments)

        # Join route location adjustments (season-wide, so join on player only)
        contributions = contributions.join(
            all_route_adj.select(['player_id', 'player_name', 'route_location_adjustment']),
            on=['player_id', 'player_name'],
            how='left'
        )
    else:
        contributions = contributions.with_columns([
            pl.lit(1.0).alias('route_location_adjustment')
        ])

    # Fill missing route location adjustments with 1.0 (neutral)
    contributions = contributions.with_columns([
        pl.col('route_location_adjustment').fill_null(1.0)
    ])

    # Calculate per-week turnover attribution penalties (Phase 6.4) for QB/RB
    # Process each position separately since turnover types differ by position
    turnover_penalties = []

    for position in unique_players['position'].unique():
        position_contribs = contributions.filter(pl.col('position') == position)
        if position_contribs.height > 0:
            turnover_df = calculate_turnover_attribution_penalties_batch(position_contribs, pbp_data, year, position)
            turnover_penalties.append(turnover_df)

    # Combine all turnover penalties
    if turnover_penalties:
        all_turnover_penalties = pl.concat(turnover_penalties)

        # Join turnover penalties (per-week, so join on player + team + week)
        contributions = contributions.join(
            all_turnover_penalties.select(['player_id', 'player_name', 'team', 'week', 'turnover_penalty']),
            on=['player_id', 'player_name', 'team', 'week'],
            how='left'
        )
    else:
        contributions = contributions.with_columns([
            pl.lit(0.0).alias('turnover_penalty')
        ])

    # Fill missing turnover penalties with 0 (no turnovers)
    contributions = contributions.with_columns([
        pl.col('turnover_penalty').fill_null(0.0)
    ])

    # Apply combined Phase 4 multipliers to player_overall_contribution
    # Includes Phase 6.1 success rate + Phase 6.3 route location
    contributions = contributions.with_columns([
        (pl.col('player_overall_contribution') *
         pl.col('catch_rate_adjustment') *
         pl.col('blocking_adjustment') *
         pl.col('separation_adjustment') *
         pl.col('penalty_adjustment') *
         pl.col('success_rate_adjustment') *
         pl.col('route_location_adjustment')).alias('player_overall_contribution')
    ])

    # Apply Phase 6.4 turnover penalties (additive, after all multipliers)
    contributions = contributions.with_columns([
        (pl.col('player_overall_contribution') + pl.col('turnover_penalty')).alias('player_overall_contribution')
    ])

    logger.info(f"Phase 4 adjustments (including Phase 6.1-6.4: success rate + route location + turnovers) applied to {len(unique_players)} players")

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
