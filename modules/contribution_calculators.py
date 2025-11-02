"""
Contribution Calculators Module

Core calculation logic for player contributions, difficulty adjustments, and opponent adjustments.
Extracted from main.py to improve modularity and maintainability.

Functions:
- calculate_offensive_shares(): Core RB/WR/TE contribution calculation with situational adjustments
- calculate_qb_contribution_from_pbp(): QB-specific contribution with FTN contextual adjustments
- calculate_average_difficulty(): Difficulty multiplier calculations across positions
- calculate_difficulty_context(): Detailed difficulty context metrics (box count, coverage)
- adjust_for_opponent(): Opponent strength adjustments using rolling defensive performance
- calculate_defensive_stats(): Helper for opponent adjustments
- adjust_for_game_situation(): Situational adjustments from PBP context
- get_personnel_multiplier(): Personnel group multiplier lookup
"""

import polars as pl
from pathlib import Path
from modules.logger import get_logger
from modules.play_by_play import PlayByPlayProcessor
from modules.personnel_inference import PersonnelInference
from modules.constants import COMBINED_METRICS, SKILL_POSITIONS

# Initialize module components
logger = get_logger(__name__)
pbp_processor = PlayByPlayProcessor()
personnel_inferencer = PersonnelInference()


def adjust_for_game_situation(df: pl.DataFrame, year: int = None, week: int = None, team: str = None) -> pl.DataFrame:
    """Apply situational adjustments to stats based on game context from play-by-play data."""

    # If we have the necessary info, try to get PBP data
    if year is not None and week is not None and team is not None:
        try:
            pbp_data = pbp_processor.get_situational_context(year, week, team)

            if pbp_data is not None:
                # Aggregate PBP multipliers to game level (average of all plays)
                game_multipliers = pbp_data.select([
                    pl.col("field_position_multiplier").mean().alias("redzone_multiplier"),
                    pl.col("score_multiplier").mean().alias("game_state_multiplier"),
                    pl.col("time_multiplier").mean(),
                    pl.col("down_multiplier").mean()
                ])

                # Add these as constants to the dataframe
                return df.with_columns([
                    pl.lit(game_multipliers["redzone_multiplier"][0]).alias("redzone_multiplier"),
                    pl.lit(game_multipliers["game_state_multiplier"][0]).alias("game_state_multiplier"),
                    pl.lit(game_multipliers["time_multiplier"][0]).alias("time_multiplier"),
                    pl.lit(game_multipliers["down_multiplier"][0]).alias("down_multiplier")
                ])
        except Exception as e:
            logger.debug(f"Could not load PBP data for {team} week {week} {year}: {str(e)}")

    # Fallback to default multipliers if PBP data not available
    return df.with_columns([
        pl.lit(1.0).alias("redzone_multiplier"),
        pl.lit(1.0).alias("game_state_multiplier"),
        pl.lit(1.0).alias("time_multiplier"),
        pl.lit(1.0).alias("down_multiplier")
    ])


def calculate_defensive_stats(team_stats: pl.DataFrame) -> pl.DataFrame:
    """Calculate comprehensive defensive stats by matching opponent's offensive performance.

    For each team's game, we look up what their opponent did to determine what the defense allowed.
    Includes yards, points, sacks, pressures, INTs, TFLs, etc.
    """
    # Calculate points scored for each team-week
    team_stats = team_stats.with_columns([
        # Total points = (passing_tds + rushing_tds + def_tds + special_teams_tds) * 7 + (pat_made * 1) + (fg_made * 3)
        # Simplified: TDs * 7 (assuming PATs are usually successful) + field goals
        ((pl.col('passing_tds').fill_null(0) +
          pl.col('rushing_tds').fill_null(0) +
          pl.col('def_tds').fill_null(0) +
          pl.col('special_teams_tds').fill_null(0)) * 7 +
         pl.col('fg_made').fill_null(0) * 3 +
         pl.col('pat_made').fill_null(0)).alias('points_scored')
    ])

    # Create opponent lookup: team X's opponent_team Y's offensive stats = what team X's defense allowed
    opponent_offense = team_stats.select([
        pl.col('week'),
        pl.col('team').alias('opponent_team'),
        # Points and yards allowed (offensive production = defensive stats allowed)
        pl.col('points_scored').alias('opponent_points_scored'),
        pl.col('passing_yards').alias('opponent_passing_yards'),
        pl.col('rushing_yards').alias('opponent_rushing_yards'),
        pl.col('passing_tds').alias('opponent_passing_tds'),
        pl.col('rushing_tds').alias('opponent_rushing_tds'),
        # Sacks/pressures suffered by opponent (offensive stat) = what defense generated
        pl.col('sacks_suffered').fill_null(0).alias('opponent_sacks_suffered'),
        # Turnovers by opponent offense = what defense forced
        pl.col('passing_interceptions').fill_null(0).alias('opponent_interceptions_thrown'),
        pl.col('rushing_fumbles_lost').fill_null(0).alias('opponent_rush_fumbles_lost'),
        pl.col('receiving_fumbles_lost').fill_null(0).alias('opponent_rec_fumbles_lost')
    ])

    # Join back to get what the opponent's offense did = what this team's defense allowed/generated
    defensive_stats = team_stats.join(
        opponent_offense,
        on=['week', 'opponent_team'],
        how='left'
    ).with_columns([
        # What defense allowed - cast to Float64 to ensure numeric type
        pl.col('opponent_points_scored').cast(pl.Float64).fill_null(0.0).alias('points_allowed'),
        pl.col('opponent_passing_yards').cast(pl.Float64).fill_null(0.0).alias('passing_yards_allowed'),
        pl.col('opponent_rushing_yards').cast(pl.Float64).fill_null(0.0).alias('rushing_yards_allowed'),
        pl.col('opponent_passing_tds').cast(pl.Float64).fill_null(0.0).alias('passing_tds_allowed'),
        pl.col('opponent_rushing_tds').cast(pl.Float64).fill_null(0.0).alias('rushing_tds_allowed'),
        # What defense generated (sacks on opponent QB)
        pl.col('opponent_sacks_suffered').cast(pl.Float64).fill_null(0.0).alias('sacks_generated'),
        # Turnovers forced
        pl.col('opponent_interceptions_thrown').cast(pl.Float64).fill_null(0.0).alias('interceptions_forced'),
        (pl.col('opponent_rush_fumbles_lost').cast(pl.Float64).fill_null(0.0) +
         pl.col('opponent_rec_fumbles_lost').cast(pl.Float64).fill_null(0.0)).alias('fumbles_forced')
    ])

    # Add direct defensive stats from team's own defensive columns - cast to Float64
    defensive_stats = defensive_stats.with_columns([
        pl.col('def_sacks').cast(pl.Float64).fill_null(0.0).alias('def_sacks_recorded'),
        pl.col('def_qb_hits').cast(pl.Float64).fill_null(0.0).alias('def_qb_hits_recorded'),
        pl.col('def_interceptions').cast(pl.Float64).fill_null(0.0).alias('def_interceptions_recorded'),
        pl.col('def_tackles_for_loss').cast(pl.Float64).fill_null(0.0).alias('def_tfl_recorded')
    ])

    return defensive_stats


def adjust_for_opponent(df: pl.DataFrame, opponent_stats: pl.DataFrame) -> pl.DataFrame:
    """Apply opponent strength adjustments based on rolling defensive performance.

    Uses rolling average of points/yards allowed through the week the game was played,
    providing a more accurate representation of defensive strength at that point in time.
    """
    # First, calculate defensive stats (points/yards allowed) from offensive stats
    opponent_with_defense = calculate_defensive_stats(opponent_stats)

    # Calculate rolling averages for each team through each week
    # Sort by team and week to ensure proper rolling calculation
    opponent_rolling = opponent_with_defense.sort(['team', 'week'])

    # Calculate game weights: ramp from 0.25 to 1.0 over first 4 games
    # This downweights early-season volatility
    opponent_rolling = opponent_rolling.with_columns([
        pl.col('week').cum_count().over('team').alias('game_number')
    ]).with_columns([
        # Weight formula: min(game_number * 0.25, 1.0)
        pl.when(pl.col('game_number') >= 4)
          .then(1.0)
          .otherwise(pl.col('game_number') * 0.25)
          .alias('game_weight')
    ])

    # Calculate weighted cumulative stats (sum of stat * weight for each team)
    opponent_rolling = opponent_rolling.with_columns([
        # Weighted cumulative sums
        (pl.col('points_allowed').fill_null(0) * pl.col('game_weight')).cum_sum().over('team').alias('weighted_cum_points_allowed'),
        (pl.col('passing_yards_allowed').fill_null(0) * pl.col('game_weight')).cum_sum().over('team').alias('weighted_cum_pass_yards_allowed'),
        (pl.col('rushing_yards_allowed').fill_null(0) * pl.col('game_weight')).cum_sum().over('team').alias('weighted_cum_rush_yards_allowed'),
        (pl.col('passing_tds_allowed').fill_null(0) * pl.col('game_weight')).cum_sum().over('team').alias('weighted_cum_pass_tds_allowed'),
        (pl.col('rushing_tds_allowed').fill_null(0) * pl.col('game_weight')).cum_sum().over('team').alias('weighted_cum_rush_tds_allowed'),
        (pl.col('def_sacks_recorded').fill_null(0) * pl.col('game_weight')).cum_sum().over('team').alias('weighted_cum_sacks'),
        (pl.col('def_qb_hits_recorded').fill_null(0) * pl.col('game_weight')).cum_sum().over('team').alias('weighted_cum_qb_hits'),
        (pl.col('def_interceptions_recorded').fill_null(0) * pl.col('game_weight')).cum_sum().over('team').alias('weighted_cum_interceptions'),
        (pl.col('def_tfl_recorded').fill_null(0) * pl.col('game_weight')).cum_sum().over('team').alias('weighted_cum_tfls'),
        # Cumulative sum of weights (denominator)
        pl.col('game_weight').cum_sum().over('team').alias('cum_weight')
    ]).with_columns([
        # Calculate weighted rolling averages (weighted sum / sum of weights)
        (pl.col('weighted_cum_points_allowed') / pl.col('cum_weight')).alias('rolling_ppg_allowed'),
        (pl.col('weighted_cum_pass_yards_allowed') / pl.col('cum_weight')).alias('rolling_pass_ypg_allowed'),
        (pl.col('weighted_cum_rush_yards_allowed') / pl.col('cum_weight')).alias('rolling_rush_ypg_allowed'),
        (pl.col('weighted_cum_pass_tds_allowed') / pl.col('cum_weight')).alias('rolling_pass_tds_allowed'),
        (pl.col('weighted_cum_rush_tds_allowed') / pl.col('cum_weight')).alias('rolling_rush_tds_allowed'),
        # Pass defense rates
        (pl.col('weighted_cum_sacks') / pl.col('cum_weight')).alias('rolling_sacks_pg'),
        (pl.col('weighted_cum_qb_hits') / pl.col('cum_weight')).alias('rolling_qb_hits_pg'),
        (pl.col('weighted_cum_interceptions') / pl.col('cum_weight')).alias('rolling_ints_pg'),
        # Run defense rates
        (pl.col('weighted_cum_tfls') / pl.col('cum_weight')).alias('rolling_tfls_pg')
    ])

    # Calculate league averages for this season (using same rolling approach)
    league_rolling_avg = opponent_rolling.group_by('week').agg([
        pl.col('rolling_ppg_allowed').mean().alias('league_avg_ppg'),
        pl.col('rolling_pass_ypg_allowed').mean().alias('league_avg_pass_ypg'),
        pl.col('rolling_rush_ypg_allowed').mean().alias('league_avg_rush_ypg'),
        pl.col('rolling_pass_tds_allowed').mean().alias('league_avg_pass_tds'),
        pl.col('rolling_rush_tds_allowed').mean().alias('league_avg_rush_tds'),
        pl.col('rolling_sacks_pg').mean().alias('league_avg_sacks'),
        pl.col('rolling_qb_hits_pg').mean().alias('league_avg_qb_hits'),
        pl.col('rolling_ints_pg').mean().alias('league_avg_ints'),
        pl.col('rolling_tfls_pg').mean().alias('league_avg_tfls')
    ])

    # Join league averages back
    opponent_rolling = opponent_rolling.join(league_rolling_avg, on='week', how='left')

    # Join with the main dataframe
    # Match on opponent team AND week to get the defense's performance through that week
    df_with_opponent = df.join(
        opponent_rolling.select([
            'team', 'week',
            'rolling_ppg_allowed', 'rolling_pass_ypg_allowed', 'rolling_rush_ypg_allowed',
            'rolling_pass_tds_allowed', 'rolling_rush_tds_allowed',
            'rolling_sacks_pg', 'rolling_qb_hits_pg', 'rolling_ints_pg', 'rolling_tfls_pg',
            'league_avg_ppg', 'league_avg_pass_ypg', 'league_avg_rush_ypg',
            'league_avg_pass_tds', 'league_avg_rush_tds',
            'league_avg_sacks', 'league_avg_qb_hits', 'league_avg_ints', 'league_avg_tfls'
        ]),
        left_on=['opponent_team', 'week'],
        right_on=['team', 'week'],
        how='left'
    )

    # Calculate individual component multipliers
    # Better defense (lower allowed, higher forced) = higher multiplier
    # Formula: league_avg / opponent_avg for "allowed" stats (yards, TDs)
    # Formula: opponent_avg / league_avg for "generated" stats (sacks, INTs, TFLs)
    # Add small epsilon to denominators to prevent division by zero
    # Cap multipliers at 0.5-2.5 range to balance elite defense recognition with reasonable score scaling
    df_with_opponent = df_with_opponent.with_columns([
        # Pass defense components (lower is better for defense = harder for offense)
        (pl.col('league_avg_pass_ypg') / (pl.col('rolling_pass_ypg_allowed') + 0.1)).fill_null(1.0).fill_nan(1.0).clip(0.5, 2.5).alias('pass_yards_mult'),
        (pl.col('rolling_sacks_pg') / (pl.col('league_avg_sacks') + 0.01)).fill_null(1.0).fill_nan(1.0).clip(0.5, 2.5).alias('sacks_mult'),
        (pl.col('rolling_qb_hits_pg') / (pl.col('league_avg_qb_hits') + 0.01)).fill_null(1.0).fill_nan(1.0).clip(0.5, 2.5).alias('qb_hits_mult'),
        (pl.col('rolling_ints_pg') / (pl.col('league_avg_ints') + 0.01)).fill_null(1.0).fill_nan(1.0).clip(0.5, 2.5).alias('ints_mult'),
        (pl.col('league_avg_pass_tds') / (pl.col('rolling_pass_tds_allowed') + 0.05)).fill_null(1.0).fill_nan(1.0).clip(0.5, 2.5).alias('pass_td_mult'),
        # Run defense components
        (pl.col('league_avg_rush_ypg') / (pl.col('rolling_rush_ypg_allowed') + 0.1)).fill_null(1.0).fill_nan(1.0).clip(0.5, 2.5).alias('rush_yards_mult'),
        (pl.col('rolling_tfls_pg') / (pl.col('league_avg_tfls') + 0.01)).fill_null(1.0).fill_nan(1.0).clip(0.5, 2.5).alias('tfl_mult'),
        (pl.col('league_avg_rush_tds') / (pl.col('rolling_rush_tds_allowed') + 0.05)).fill_null(1.0).fill_nan(1.0).clip(0.5, 2.5).alias('rush_td_mult'),
        # Overall scoring
        (pl.col('league_avg_ppg') / (pl.col('rolling_ppg_allowed') + 0.1)).fill_null(1.0).fill_nan(1.0).clip(0.5, 2.5).alias('scoring_multiplier')
    ])

    # Calculate composite pass defense quality score (weighted average)
    # Weights: yards 30%, sacks 25%, pressures 20%, INTs 15%, TDs 10%
    df_with_opponent = df_with_opponent.with_columns([
        (pl.col('pass_yards_mult') * 0.30 +
         pl.col('sacks_mult') * 0.25 +
         pl.col('qb_hits_mult') * 0.20 +
         pl.col('ints_mult') * 0.15 +
         pl.col('pass_td_mult') * 0.10).alias('pass_defense_multiplier')
    ])

    # Calculate composite rush defense quality score (weighted average)
    # Weights: yards 40%, TFLs 30%, stuffed runs 20%, TDs 10%
    df_with_opponent = df_with_opponent.with_columns([
        (pl.col('rush_yards_mult') * 0.40 +
         pl.col('tfl_mult') * 0.30 +
         pl.col('tfl_mult') * 0.20 +  # TFLs proxy for stuffed runs
         pl.col('rush_td_mult') * 0.10).alias('rush_defense_multiplier')
    ])

    return df_with_opponent


def get_personnel_multiplier(position: str, personnel_group: str) -> float:
    """
    Get position-specific personnel multiplier based on formation.

    Args:
        position: Player position (WR, RB, TE, QB)
        personnel_group: Personnel grouping like '11', '12', '21', etc.

    Returns:
        Multiplier value (0.85-1.15)
    """
    if position not in ['WR', 'RB', 'TE']:
        return 1.0  # QBs and others get neutral multiplier

    # Use PersonnelInference to get the multiplier
    # Use high confidence (0.9) since we're using actual/inferred personnel from PBP cache
    return personnel_inferencer.get_position_multiplier(personnel_group, position, confidence=0.9)


def calculate_offensive_shares(team_stats: pl.DataFrame, player_stats: pl.DataFrame, metric: str,
                             opponent_stats: pl.DataFrame = None, year: int = None, week: int = None, team: str = None) -> pl.DataFrame:
    """Calculate what percentage of team total each player accounts for with situational and opponent adjustments."""
    # Apply situational adjustments first.
    # If a year is provided (season-level calculation), build a per-team-week multiplier table
    # and join it onto the team/player data so seasonal aggregates reflect per-game situational weights.
    if year is not None:
        # Build multipliers for each (team, week) present in team_stats
        pairs = team_stats.select(['team', 'week']).unique().to_dicts()
        multipliers = []
        for p in pairs:
            t = p['team']
            w = int(p['week'])
            try:
                pbp_data = pbp_processor.get_situational_context(year, w, t)
                if pbp_data is not None and pbp_data.height > 0:
                    gm = pbp_data.select([
                        pl.col("field_position_multiplier").mean().alias("redzone_multiplier"),
                        pl.col("score_multiplier").mean().alias("game_state_multiplier"),
                        pl.col("time_multiplier").mean().alias("time_multiplier"),
                        pl.col("down_multiplier").mean().alias("down_multiplier"),
                        pl.col("personnel_group").mode().first().alias("personnel_group"),  # Most common personnel
                        pl.col("defense_coverage_type").mode().first().alias("coverage_type"),  # Most common coverage
                        pl.col("defenders_in_box").mean().alias("avg_defenders_in_box")  # Average box count
                    ])
                    multipliers.append({
                        'team': t,
                        'week': w,
                        'redzone_multiplier': float(gm[0, 'redzone_multiplier']),
                        'game_state_multiplier': float(gm[0, 'game_state_multiplier']),
                        'time_multiplier': float(gm[0, 'time_multiplier']),
                        'down_multiplier': float(gm[0, 'down_multiplier']),
                        'personnel_group': str(gm[0, 'personnel_group']) if gm[0, 'personnel_group'] is not None else '11',
                        'coverage_type': str(gm[0, 'coverage_type']) if gm[0, 'coverage_type'] is not None else 'COVER_2',
                        'avg_defenders_in_box': float(gm[0, 'avg_defenders_in_box']) if gm[0, 'avg_defenders_in_box'] is not None else 6.0
                    })
                else:
                    multipliers.append({
                        'team': t, 'week': w,
                        'redzone_multiplier': 1.0, 'game_state_multiplier': 1.0,
                        'time_multiplier': 1.0, 'down_multiplier': 1.0,
                        'personnel_group': '11',  # Default to most common personnel
                        'coverage_type': 'COVER_2',
                        'avg_defenders_in_box': 6.0
                    })
            except Exception:
                multipliers.append({
                    'team': t, 'week': w,
                    'redzone_multiplier': 1.0, 'game_state_multiplier': 1.0,
                    'time_multiplier': 1.0, 'down_multiplier': 1.0,
                    'personnel_group': '11',  # Default to most common personnel
                    'coverage_type': 'COVER_2',
                    'avg_defenders_in_box': 6.0
                })

        if len(multipliers) > 0:
            mult_df = pl.DataFrame(multipliers)
            # join multipliers onto team_stats and player_stats
            team_stats = team_stats.join(mult_df, on=['team', 'week'], how='left')
            player_stats = player_stats.join(mult_df, on=['team', 'week'], how='left')
        else:
            team_stats = team_stats.with_columns([
                pl.lit(1.0).alias("redzone_multiplier"), pl.lit(1.0).alias("game_state_multiplier"),
                pl.lit(1.0).alias("time_multiplier"), pl.lit(1.0).alias("down_multiplier"),
                pl.lit('11').alias("personnel_group"),
                pl.lit('COVER_2').alias("coverage_type"),
                pl.lit(6.0).alias("avg_defenders_in_box")
            ])
            player_stats = player_stats.with_columns([
                pl.lit(1.0).alias("redzone_multiplier"), pl.lit(1.0).alias("game_state_multiplier"),
                pl.lit(1.0).alias("time_multiplier"), pl.lit(1.0).alias("down_multiplier"),
                pl.lit('11').alias("personnel_group"),
                pl.lit('COVER_2').alias("coverage_type"),
                pl.lit(6.0).alias("avg_defenders_in_box")
            ])
    else:
        # No year provided -> raw values: set multipliers to 1.0
        team_stats = team_stats.with_columns([
            pl.lit(1.0).alias("redzone_multiplier"), pl.lit(1.0).alias("game_state_multiplier"),
            pl.lit(1.0).alias("time_multiplier"), pl.lit(1.0).alias("down_multiplier"),
            pl.lit('11').alias("personnel_group"),
            pl.lit('COVER_2').alias("coverage_type"),
            pl.lit(6.0).alias("avg_defenders_in_box")
        ])
        player_stats = player_stats.with_columns([
            pl.lit(1.0).alias("redzone_multiplier"), pl.lit(1.0).alias("game_state_multiplier"),
            pl.lit(1.0).alias("time_multiplier"), pl.lit(1.0).alias("down_multiplier"),
            pl.lit('11').alias("personnel_group"),
            pl.lit('COVER_2').alias("coverage_type"),
            pl.lit(6.0).alias("avg_defenders_in_box")
        ])

    # Apply opponent adjustments if available
    if opponent_stats is not None:
        team_stats = adjust_for_opponent(team_stats, opponent_stats)
        player_stats = adjust_for_opponent(player_stats, opponent_stats)

    if metric in COMBINED_METRICS:
        # For combined metrics, sum the component metrics first
        combined_info = COMBINED_METRICS[metric]

        if metric == 'overall_contribution':
            # For overall contribution, apply weights with situational and opponent adjustments
            # Process team_stats
            weighted_cols_team = []
            max_possible_score = 0  # Calculate theoretical max for normalization
            for m in combined_info['metrics']:
                base_weight = combined_info['weights'][m]

                # Apply situational multipliers
                situational_weight = (
                    pl.col(m).fill_null(0) * base_weight *
                    pl.col("redzone_multiplier") *
                    pl.col("game_state_multiplier") *
                    pl.col("time_multiplier") *
                    pl.col("down_multiplier")
                )

                # Apply opponent adjustments if available
                if any(col in team_stats.columns for col in ["pass_defense_multiplier", "rush_defense_multiplier"]):
                    if m in ['receiving_yards', 'receiving_tds', 'receptions', 'targets', 'passing_yards', 'passing_tds', 'completions', 'attempts']:
                        situational_weight = situational_weight * pl.col("pass_defense_multiplier")
                    elif m in ['rushing_yards', 'rushing_tds', 'carries']:
                        situational_weight = situational_weight * pl.col("rush_defense_multiplier")

                weighted_cols_team.append(situational_weight)

                # Calculate max possible score with all bonuses
                max_values = {
                    'receiving_yards': 400, 'rushing_yards': 400,
                    'receiving_tds': 4, 'rushing_tds': 4,
                    'receptions': 15, 'targets': 20, 'carries': 20
                }
                # Maximum multiplier combination (all situational bonuses * max opponent adjustment)
                max_multiplier = 1.5 * 1.3 * 1.4 * 1.5 * 1.5  # Product of max possible multipliers
                max_possible_score += max_values[m] * base_weight * max_multiplier

            # Sum all weighted contributions for team
            team_stats = team_stats.with_columns([
                pl.sum_horizontal(weighted_cols_team).alias(metric)
            ])

            # Process player_stats
            weighted_cols_player = []
            for m in combined_info['metrics']:
                base_weight = combined_info['weights'][m]

                # Apply situational multipliers
                situational_weight = (
                    pl.col(m).fill_null(0) * base_weight *
                    pl.col("redzone_multiplier") *
                    pl.col("game_state_multiplier") *
                    pl.col("time_multiplier") *
                    pl.col("down_multiplier")
                )

                # Apply opponent adjustments if available
                if any(col in player_stats.columns for col in ["pass_defense_multiplier", "rush_defense_multiplier"]):
                    if m in ['receiving_yards', 'receiving_tds', 'receptions', 'targets', 'passing_yards', 'passing_tds', 'completions', 'attempts']:
                        situational_weight = situational_weight * pl.col("pass_defense_multiplier")
                    elif m in ['rushing_yards', 'rushing_tds', 'carries']:
                        situational_weight = situational_weight * pl.col("rush_defense_multiplier")

                weighted_cols_player.append(situational_weight)

            # Sum all weighted contributions for players
            player_stats = player_stats.with_columns([
                pl.sum_horizontal(weighted_cols_player).alias(metric)
            ])

            # Apply position-specific personnel multipliers (Phase 1)
            # Only apply if we have position and personnel_group columns
            if 'position' in player_stats.columns and 'personnel_group' in player_stats.columns:
                # Calculate personnel multiplier for each player based on their position
                player_stats = player_stats.with_columns([
                    pl.struct(['position', 'personnel_group'])
                      .map_elements(
                          lambda row: get_personnel_multiplier(row['position'], row['personnel_group']),
                          return_dtype=pl.Float64
                      )
                      .alias('personnel_multiplier')
                ])

                # Apply personnel multiplier to the overall contribution
                player_stats = player_stats.with_columns([
                    (pl.col(metric) * pl.col('personnel_multiplier')).alias(metric)
                ])

                # Apply special case combo penalties (Phase 1 Enhancement)
                # These are additional penalties for particularly easy situations
                if 'coverage_type' in player_stats.columns:
                    player_stats = player_stats.with_columns([
                        pl.when(
                            # RB receiving in 10 personnel (spread checkdowns)
                            # Applies to RBs in spread formations (likely checkdowns)
                            (pl.col('position') == 'RB') &
                            (pl.col('personnel_group') == '10')
                        ).then(pl.col(metric) * 0.85)
                        .when(
                            # WR in prevent defense + 10 personnel (double easy - garbage time + spread)
                            (pl.col('position') == 'WR') &
                            (pl.col('coverage_type') == 'PREVENT') &
                            (pl.col('personnel_group') == '10')
                        ).then(pl.col(metric) * 0.80)
                        .when(
                            # TE receiving in heavy personnel (13/22 - wide open mismatches)
                            (pl.col('position') == 'TE') &
                            (pl.col('personnel_group').is_in(['13', '22']))
                        ).then(pl.col(metric) * 0.85)
                        .otherwise(pl.col(metric))
                        .alias(metric)
                    ])
        else:
            # For simple combined metrics, just sum the components
            team_stats = team_stats.with_columns([
                pl.sum_horizontal([pl.col(m).fill_null(0) for m in combined_info['metrics']]).alias(metric)
            ])
            player_stats = player_stats.with_columns([
                pl.sum_horizontal([pl.col(m).fill_null(0) for m in combined_info['metrics']]).alias(metric)
            ])

    # Group by team and week to get team totals
    team_totals = team_stats.group_by(['team', 'week']).agg([
        pl.col(metric).sum().alias(f'team_{metric}')
    ])

    # Group by player, team, and week to get player totals
    player_totals = player_stats.group_by(['player_id', 'player_name', 'team', 'week']).agg([
        pl.col(metric).sum().alias(f'player_{metric}')
    ])

    # Join and calculate percentages
    result = player_totals.join(
        team_totals,
        on=['team', 'week']
    ).with_columns([
        # Handle division by zero - if team total is 0, set share to 0
        pl.when(pl.col(f'team_{metric}') == 0)
        .then(0.0)
        .otherwise(pl.col(f'player_{metric}') / pl.col(f'team_{metric}') * 100)
        .alias(f'{metric}_share')
    ])

    return result


def calculate_qb_contribution_from_pbp(year: int, qb_name: str) -> float:
    """
    Calculate QB contribution from play-by-play data with contextual penalties and rewards.

    Positive contributions:
    - Passing yards, TDs, completions, rushing yards/TDs (base weights)
    - Bonus for completions under pressure (1.4x completion, 1.2x yards) when data available
    - Bonus for out of pocket completions (+3 points) when FTN data available (2022+)

    Contextual adjustments (FTN data, 2022+):
    - Play action: -10% (easier reads)
    - Out of pocket: +3 pts per completion (improvisation value)
    - Blitz: -2 pts per completion (defense took risk)
    - Screen pass: -15% (high completion rate, low value)

    Negative penalties (with situation modifiers):
    - Interceptions: -35 base
    - Sacks: -10 base
    - Fumbles lost: -35 base

    Situation modifiers adjust penalties based on:
    - Field position (own territory worse)
    - Score differential (close games more critical)
    - Time remaining (final 2 min more critical)
    - Down & distance (3rd/4th down less harsh)
    """
    pbp_data = pbp_processor.load_pbp_data(year)
    if pbp_data is None:
        return 0.0

    # Load FTN charting data if available (2022+)
    from modules.ftn_cache_builder import load_ftn_cache, FTN_START_YEAR
    ftn_data = None
    if year >= FTN_START_YEAR:
        ftn_data = load_ftn_cache(year)
        if ftn_data is not None:
            # Ensure both join keys are Int64 for consistent types (handles Float64/Int32/Int64 sources)
            if 'play_id' in pbp_data.columns:
                pbp_data = pbp_data.with_columns([
                    pl.col('play_id').cast(pl.Int64, strict=False)
                ])
            if 'nflverse_play_id' in ftn_data.columns:
                ftn_data = ftn_data.with_columns([
                    pl.col('nflverse_play_id').cast(pl.Int64, strict=False)
                ])
            # Join FTN data with PBP using consistent Int64 types
            pbp_data = pbp_data.join(
                ftn_data,
                left_on=['game_id', 'play_id'],
                right_on=['nflverse_game_id', 'nflverse_play_id'],
                how='left'
            )
            # Log only once per year (not per QB)
            # logger.info(f"FTN data joined for {year} QB analysis ({len(ftn_data)} charted plays)")

    # Check if pressure data is available (2024 and earlier have ~96% coverage)
    has_pressure_data = pbp_data['was_pressure'].null_count() < len(pbp_data) * 0.5
    has_ftn_data = ftn_data is not None

    weights = COMBINED_METRICS['qb_contribution']['weights']
    contribution = 0.0

    # Get all plays for this QB
    qb_plays = pbp_data.filter(pl.col('passer_player_name') == qb_name)

    if len(qb_plays) == 0:
        return 0.0

    # === POSITIVE CONTRIBUTIONS ===

    # Passing yards - with pressure bonus and FTN contextual adjustments
    if has_pressure_data and has_ftn_data:
        # Process each pass play with full context
        pass_plays = qb_plays.filter(pl.col('passing_yards').is_not_null())

        for play in pass_plays.iter_rows(named=True):
            yards = play.get('passing_yards', 0)
            base_value = yards * weights['passing_yards']

            # Pressure bonus (1.2x)
            if play.get('was_pressure') == 1:
                base_value *= 1.2

            # FTN contextual adjustments
            ftn_modifier = 1.0

            # Play action: -10%
            if play.get('is_play_action') == True:
                ftn_modifier *= 0.9

            # Screen pass: -15%
            if play.get('is_screen_pass') == True:
                ftn_modifier *= 0.85

            contribution += base_value * ftn_modifier

    elif has_pressure_data:
        # Pressure bonus only (no FTN data)
        clean_yards = qb_plays.filter(
            (pl.col('passing_yards').is_not_null()) &
            ((pl.col('was_pressure') == 0) | (pl.col('was_pressure').is_null()))
        )['passing_yards'].sum()

        pressure_yards = qb_plays.filter(
            (pl.col('passing_yards').is_not_null()) &
            (pl.col('was_pressure') == 1)
        )['passing_yards'].sum()

        contribution += (clean_yards or 0) * weights['passing_yards']
        contribution += (pressure_yards or 0) * weights['passing_yards'] * 1.2

    elif has_ftn_data:
        # FTN adjustments only (no pressure data)
        pass_plays = qb_plays.filter(pl.col('passing_yards').is_not_null())

        for play in pass_plays.iter_rows(named=True):
            yards = play.get('passing_yards', 0)
            base_value = yards * weights['passing_yards']

            ftn_modifier = 1.0

            if play.get('is_play_action') == True:
                ftn_modifier *= 0.9

            if play.get('is_screen_pass') == True:
                ftn_modifier *= 0.85

            contribution += base_value * ftn_modifier

    else:
        # No pressure or FTN data - standard calculation
        total_yards = qb_plays.filter(pl.col('passing_yards').is_not_null())['passing_yards'].sum()
        contribution += (total_yards or 0) * weights['passing_yards']

    # Passing TDs
    passing_tds = qb_plays.filter(pl.col('pass_touchdown') == 1).height
    contribution += passing_tds * weights['passing_tds']

    # Completions - with pressure bonus and FTN contextual adjustments
    if has_pressure_data and has_ftn_data:
        # Process each completion with full context
        completions = qb_plays.filter(pl.col('complete_pass') == 1)

        for comp in completions.iter_rows(named=True):
            base_value = weights['completions']

            # Pressure bonus (1.4x)
            if comp.get('was_pressure') == 1:
                base_value *= 1.4

            # FTN contextual adjustments
            ftn_modifier = 1.0
            ftn_bonus = 0

            # Play action: -10% (easier reads)
            if comp.get('is_play_action') == True:
                ftn_modifier *= 0.9

            # Screen pass: -15% (high completion rate, low difficulty)
            if comp.get('is_screen_pass') == True:
                ftn_modifier *= 0.85

            # Out of pocket: +3 points (improvisation bonus)
            if comp.get('is_qb_out_of_pocket') == True:
                ftn_bonus += 3.0

            # Blitz: -2 points (defense took risk, easier read)
            n_blitzers = comp.get('n_blitzers')
            if n_blitzers is not None and n_blitzers >= 5:
                ftn_bonus -= 2.0

            contribution += (base_value * ftn_modifier) + ftn_bonus

    elif has_pressure_data:
        # Pressure bonus only (no FTN data)
        clean_completions = qb_plays.filter(
            (pl.col('complete_pass') == 1) &
            ((pl.col('was_pressure') == 0) | (pl.col('was_pressure').is_null()))
        ).height

        pressure_completions = qb_plays.filter(
            (pl.col('complete_pass') == 1) &
            (pl.col('was_pressure') == 1)
        ).height

        contribution += clean_completions * weights['completions']
        contribution += pressure_completions * weights['completions'] * 1.4

    elif has_ftn_data:
        # FTN adjustments only (no pressure data)
        completions = qb_plays.filter(pl.col('complete_pass') == 1)

        for comp in completions.iter_rows(named=True):
            base_value = weights['completions']

            ftn_modifier = 1.0
            ftn_bonus = 0

            if comp.get('is_play_action') == True:
                ftn_modifier *= 0.9

            if comp.get('is_screen_pass') == True:
                ftn_modifier *= 0.85

            if comp.get('is_qb_out_of_pocket') == True:
                ftn_bonus += 3.0

            n_blitzers = comp.get('n_blitzers')
            if n_blitzers is not None and n_blitzers >= 5:
                ftn_bonus -= 2.0

            contribution += (base_value * ftn_modifier) + ftn_bonus

    else:
        # No pressure or FTN data - standard calculation
        completions = qb_plays.filter(pl.col('complete_pass') == 1).height
        contribution += completions * weights['completions']

    # Attempts (penalty for incompletions)
    attempts = qb_plays.filter(pl.col('pass_attempt') == 1).height
    contribution += attempts * weights['attempts']

    # Rushing yards and TDs
    rushing_yards = qb_plays.filter(
        (pl.col('rusher_player_name') == qb_name) &
        (pl.col('rushing_yards').is_not_null())
    )['rushing_yards'].sum()
    contribution += (rushing_yards or 0) * weights['rushing_yards']

    rushing_tds = qb_plays.filter(
        (pl.col('rusher_player_name') == qb_name) &
        (pl.col('rush_touchdown') == 1)
    ).height
    contribution += rushing_tds * weights['rushing_tds']

    # === NEGATIVE PENALTIES ===

    # Interceptions with contextual modifiers
    ints = qb_plays.filter(pl.col('interception') == 1)
    for int_play in ints.iter_rows(named=True):
        base_penalty = -50.0

        # Field position modifier (worse in own territory)
        yardline = int_play.get('yardline_100', 50)
        if yardline > 65:  # Own 35 or deeper
            field_mod = 1.5
        elif yardline < 35:  # Opponent 35 or closer (red zone area)
            field_mod = 0.8
        else:
            field_mod = 1.0

        # Score differential modifier
        score_diff = abs(int_play.get('score_differential', 0))
        if score_diff <= 8:  # One score game
            score_mod = 1.2
        elif score_diff >= 14:  # Losing by 2+ scores (garbage time)
            score_mod = 0.8
        else:
            score_mod = 1.0

        # Time remaining modifier
        seconds_remaining = int_play.get('game_seconds_remaining', 1800)
        if seconds_remaining < 120:  # Final 2 minutes
            time_mod = 1.3
        else:
            time_mod = 1.0

        # Down modifier (3rd/4th down attempts less harsh)
        down = int_play.get('down', 1)
        if down >= 3:
            down_mod = 0.7
        else:
            down_mod = 1.0

        # Apply all modifiers
        final_penalty = base_penalty * field_mod * score_mod * time_mod * down_mod
        contribution += final_penalty

    # Sacks with contextual modifiers
    sacks = qb_plays.filter(pl.col('sack') == 1)
    for sack_play in sacks.iter_rows(named=True):
        base_penalty = -10.0

        # Score differential modifier (less harsh when losing big)
        score_diff = sack_play.get('score_differential', 0)
        if score_diff <= -14:  # Losing by 2+ scores
            score_mod = 0.7
        else:
            score_mod = 1.0

        # Down modifier (3rd/4th down less harsh - had to try)
        down = sack_play.get('down', 1)
        if down >= 3:
            down_mod = 0.8
        else:
            down_mod = 1.0

        final_penalty = base_penalty * score_mod * down_mod
        contribution += final_penalty

    # Fumbles lost (turnovers) with contextual modifiers (same as interceptions)
    fumbles_lost = qb_plays.filter(
        (pl.col('fumble_lost') == 1) &
        ((pl.col('fumbled_1_player_name') == qb_name) | (pl.col('passer_player_name') == qb_name))
    )
    for fumble_play in fumbles_lost.iter_rows(named=True):
        base_penalty = -50.0

        # Same modifiers as interceptions
        yardline = fumble_play.get('yardline_100', 50)
        if yardline > 65:
            field_mod = 1.5
        elif yardline < 35:
            field_mod = 0.8
        else:
            field_mod = 1.0

        score_diff = abs(fumble_play.get('score_differential', 0))
        if score_diff <= 8:
            score_mod = 1.2
        elif score_diff >= 14:
            score_mod = 0.8
        else:
            score_mod = 1.0

        seconds_remaining = fumble_play.get('game_seconds_remaining', 1800)
        if seconds_remaining < 120:
            time_mod = 1.3
        else:
            time_mod = 1.0

        down = fumble_play.get('down', 1)
        if down >= 3:
            down_mod = 0.7
        else:
            down_mod = 1.0

        final_penalty = base_penalty * field_mod * score_mod * time_mod * down_mod
        contribution += final_penalty

    # Fumbles recovered by own team (still negative but less severe)
    fumbles_recovered = qb_plays.filter(
        (pl.col('fumble') == 1) &
        (pl.col('fumble_lost') == 0) &
        ((pl.col('fumbled_1_player_name') == qb_name) | (pl.col('passer_player_name') == qb_name))
    )
    for fumble_play in fumbles_recovered.iter_rows(named=True):
        # Less severe penalty since possession retained, but still lost down/yards
        base_penalty = -15.0

        # Simple modifiers - mainly just down context
        down = fumble_play.get('down', 1)
        if down >= 3:
            down_mod = 0.7  # Less harsh on 3rd/4th down
        else:
            down_mod = 1.0

        final_penalty = base_penalty * down_mod
        contribution += final_penalty

    return contribution


def calculate_average_difficulty(year: int, player_stats: pl.DataFrame) -> pl.DataFrame:
    """Calculate average difficulty multiplier for each player across all their plays.

    This combines box count, coverage, and personnel multipliers to show
    overall difficulty context each player faced.

    Returns DataFrame with:
    - player_id, player_name, position
    - avg_difficulty_multiplier (combined box × coverage × personnel)
    - total_plays
    """
    logger.info(f"Calculating average difficulty multipliers for {year}")

    # Load PBP data
    pbp_path = Path("cache/pbp") / f"pbp_{year}.parquet"
    if not pbp_path.exists():
        logger.warning(f"PBP cache not found for {year}, skipping difficulty calculation")
        return None

    pbp_data = pl.read_parquet(pbp_path)

    # Check if multiplier columns exist (only available for 2016+)
    required_cols = ['defenders_in_box_multiplier', 'coverage_multiplier']
    if not all(col in pbp_data.columns for col in required_cols):
        logger.info(f"Difficulty multiplier columns not available for {year} (data not available for this year), skipping")
        return None

    # Get unique players from player_stats
    players = player_stats.select(['player_id', 'player_name', 'position']).unique()

    difficulty_results = []

    # Process RBs (rush plays)
    rb_players = players.filter(pl.col('position') == 'RB')
    if len(rb_players) > 0:
        rb_plays = pbp_data.filter(
            (pl.col('play_type') == 'run') &
            (pl.col('rusher_player_id').is_in(rb_players['player_id'].to_list()))
        )

        if len(rb_plays) > 0:
            rb_diff = rb_plays.group_by('rusher_player_id').agg([
                # Average of (box_mult × coverage_mult) - coverage is 1.0 for rush plays
                pl.col('defenders_in_box_multiplier').mean().alias('avg_difficulty_multiplier'),
                pl.len().alias('total_plays')
            ]).rename({'rusher_player_id': 'player_id'})

            rb_diff = rb_diff.join(rb_players, on='player_id', how='left')
            difficulty_results.append(rb_diff)

    # Process QBs (pass plays + rush plays for mobile QBs)
    qb_players = players.filter(pl.col('position') == 'QB')
    if len(qb_players) > 0:
        # QB passing difficulty (coverage)
        qb_pass_plays = pbp_data.filter(
            (pl.col('play_type') == 'pass') &
            (pl.col('passer_player_id').is_in(qb_players['player_id'].to_list()))
        )

        # QB rushing difficulty (box count for scrambles/designed runs)
        qb_rush_plays = pbp_data.filter(
            (pl.col('play_type') == 'run') &
            (pl.col('rusher_player_id').is_in(qb_players['player_id'].to_list()))
        )

        # Combine both - weight by play volume (70% passing, 30% rushing for weighted average)
        qb_pass_difficulty = None
        qb_rush_difficulty = None

        if len(qb_pass_plays) > 0:
            qb_pass_difficulty = qb_pass_plays.group_by('passer_player_id').agg([
                pl.col('coverage_multiplier').mean().alias('pass_difficulty'),
                pl.len().alias('pass_plays')
            ]).rename({'passer_player_id': 'player_id'})

        if len(qb_rush_plays) > 0:
            qb_rush_difficulty = qb_rush_plays.group_by('rusher_player_id').agg([
                pl.col('defenders_in_box_multiplier').mean().alias('rush_difficulty'),
                pl.len().alias('rush_plays')
            ]).rename({'rusher_player_id': 'player_id'})

        # Combine pass and rush difficulty for QBs
        if qb_pass_difficulty is not None and qb_rush_difficulty is not None:
            qb_combined = qb_pass_difficulty.join(qb_rush_difficulty, on='player_id', how='full')
            qb_combined = qb_combined.with_columns([
                # Weighted average based on play volume
                ((pl.col('pass_difficulty').fill_null(1.0) * pl.col('pass_plays').fill_null(0) +
                  pl.col('rush_difficulty').fill_null(1.0) * pl.col('rush_plays').fill_null(0)) /
                 (pl.col('pass_plays').fill_null(0) + pl.col('rush_plays').fill_null(0))).alias('avg_difficulty_multiplier'),
                (pl.col('pass_plays').fill_null(0) + pl.col('rush_plays').fill_null(0)).alias('total_plays')
            ])
            qb_combined = qb_combined.join(qb_players, on='player_id', how='left')
            difficulty_results.append(qb_combined)
        elif qb_pass_difficulty is not None:
            # Only passing data available
            qb_pass_difficulty = qb_pass_difficulty.with_columns([
                pl.col('pass_difficulty').alias('avg_difficulty_multiplier'),
                pl.col('pass_plays').alias('total_plays')
            ])
            qb_pass_difficulty = qb_pass_difficulty.join(qb_players, on='player_id', how='left')
            difficulty_results.append(qb_pass_difficulty)
        elif qb_rush_difficulty is not None:
            # Only rushing data available (rare)
            qb_rush_difficulty = qb_rush_difficulty.with_columns([
                pl.col('rush_difficulty').alias('avg_difficulty_multiplier'),
                pl.col('rush_plays').alias('total_plays')
            ])
            qb_rush_difficulty = qb_rush_difficulty.join(qb_players, on='player_id', how='left')
            difficulty_results.append(qb_rush_difficulty)

    # Process WRs and TEs (pass plays)
    for pos in ['WR', 'TE']:
        pos_players = players.filter(pl.col('position') == pos)
        if len(pos_players) > 0:
            pass_plays = pbp_data.filter(
                (pl.col('play_type') == 'pass') &
                (pl.col('receiver_player_id').is_in(pos_players['player_id'].to_list()))
            )

            if len(pass_plays) > 0:
                pos_diff = pass_plays.group_by('receiver_player_id').agg([
                    # Average of coverage_mult (box is 1.0 for pass plays)
                    pl.col('coverage_multiplier').mean().alias('avg_difficulty_multiplier'),
                    pl.len().alias('total_plays')
                ]).rename({'receiver_player_id': 'player_id'})

                pos_diff = pos_diff.join(pos_players, on='player_id', how='left')
                difficulty_results.append(pos_diff)

    if not difficulty_results:
        return None

    # Combine all positions
    difficulty_df = pl.concat(difficulty_results, how='diagonal')

    logger.info(f"Calculated average difficulty for {len(difficulty_df)} players")
    return difficulty_df


def calculate_difficulty_context(year: int, player_stats: pl.DataFrame) -> pl.DataFrame:
    """Calculate difficulty context metrics for players from PBP data.

    Returns a DataFrame with columns:
    - player_id, player_name, position
    - avg_defenders_in_box (for RBs with rush attempts)
    - pct_vs_8plus_box (for RBs with rush attempts)
    - pct_vs_man_coverage (for WRs/TEs on pass plays)
    - avg_box_multiplier (for RBs)
    - avg_coverage_multiplier (for WRs/TEs)
    """
    logger.info(f"Calculating difficulty context for {year}")

    # Load PBP data
    pbp_path = Path("cache/pbp") / f"pbp_{year}.parquet"
    if not pbp_path.exists():
        logger.warning(f"PBP cache not found for {year}, skipping difficulty context")
        return None

    pbp_data = pl.read_parquet(pbp_path)

    # Check if required columns exist (only available for 2016+)
    required_cols = ['defenders_in_box', 'defense_coverage_type', 'defenders_in_box_multiplier', 'coverage_multiplier']
    if not all(col in pbp_data.columns for col in required_cols):
        logger.info(f"Difficulty context columns not available for {year} (data not available for this year), skipping")
        return None

    # Get unique players from player_stats
    players = player_stats.select(['player_id', 'player_name', 'position']).unique()

    difficulty_metrics = []

    for pos in ['RB', 'WR', 'TE']:
        pos_players = players.filter(pl.col('position') == pos)

        if pos == 'RB':
            # For RBs: focus on rush plays
            # Join with pbp_data on rush plays where they are the ball carrier
            rush_plays = pbp_data.filter(
                (pl.col('play_type') == 'run') &
                (pl.col('rusher_player_id').is_in(pos_players['player_id'].to_list()))
            )

            if len(rush_plays) > 0:
                rb_difficulty = rush_plays.group_by('rusher_player_id').agg([
                    pl.col('defenders_in_box').mean().alias('avg_defenders_in_box'),
                    (pl.col('defenders_in_box') >= 8).mean().alias('pct_vs_8plus_box'),
                    pl.col('defenders_in_box_multiplier').mean().alias('avg_box_multiplier'),
                    pl.len().alias('rush_attempts')
                ]).rename({'rusher_player_id': 'player_id'})

                # Join with player names/positions
                rb_difficulty = rb_difficulty.join(pos_players, on='player_id', how='left')
                difficulty_metrics.append(rb_difficulty)

        else:  # WR or TE
            # For WRs/TEs: focus on pass plays where they are targeted
            pass_plays = pbp_data.filter(
                (pl.col('play_type') == 'pass') &
                (pl.col('receiver_player_id').is_in(pos_players['player_id'].to_list()))
            )

            if len(pass_plays) > 0:
                # Classify man coverage: 2_MAN, 1_MAN, 0_MAN, COVER_1
                wr_difficulty = pass_plays.group_by('receiver_player_id').agg([
                    pl.col('defense_coverage_type').is_in(['2_MAN', '1_MAN', '0_MAN', 'COVER_1']).mean().alias('pct_vs_man_coverage'),
                    pl.col('coverage_multiplier').mean().alias('avg_coverage_multiplier'),
                    pl.len().alias('targets')
                ]).rename({'receiver_player_id': 'player_id'})

                # Join with player names/positions
                wr_difficulty = wr_difficulty.join(pos_players, on='player_id', how='left')
                difficulty_metrics.append(wr_difficulty)

    if not difficulty_metrics:
        return None

    # Combine all position difficulty metrics
    difficulty_df = pl.concat(difficulty_metrics, how='diagonal')

    logger.info(f"Calculated difficulty context for {len(difficulty_df)} players")
    return difficulty_df
