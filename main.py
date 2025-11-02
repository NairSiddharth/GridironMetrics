"""
main.py

Analyzes NFL offensive skill player shares on a weekly and seasonal basis.
Generates Markdown formatted tables showing offensive contribution percentages.
"""

from pathlib import Path
import polars as pl
from prettytable import PrettyTable, TableStyle
from modules.logger import get_logger
from modules.constants import (
    START_YEAR, END_YEAR, CACHE_DIR,
    get_team_code_for_year, get_all_team_codes_for_year,
    normalize_team_codes_in_dataframe, TEAM_RELOCATIONS
)
from modules.offensive_metrics import OffensiveMetricsCalculator
from modules.play_by_play import PlayByPlayProcessor
from modules.context_adjustments import ContextAdjustments
from modules.personnel_inference import PersonnelInference
from modules.pbp_cache_builder import build_cache
import concurrent.futures
import argparse

logger = get_logger(__name__)

# Initialize our processors
metrics_calculator = OffensiveMetricsCalculator()
pbp_processor = PlayByPlayProcessor()
context_adj = ContextAdjustments()
personnel_inferencer = PersonnelInference()

# Offensive metrics we want to track for skill positions (RB/WR/TE)
SKILL_POSITION_METRICS = {
    'rushing_yards': 'Rush Yards',
    'receiving_yards': 'Rec Yards',
    'rushing_tds': 'Rush TD',
    'receiving_tds': 'Rec TD',
    'receptions': 'Receptions',
    'targets': 'Targets',
    'carries': 'Rush Att'
}

# QB-specific metrics
QB_METRICS = {
    'passing_yards': 'Pass Yards',
    'passing_tds': 'Pass TDs',
    'completions': 'Completions',
    'attempts': 'Attempts',
    'passing_interceptions': 'Interceptions',
    'rushing_yards': 'Rush Yards',
    'rushing_tds': 'Rush TDs'
}

# Metric groupings for combined stats
COMBINED_METRICS = {
    'total_yards': {
        'metrics': ['receiving_yards', 'rushing_yards'],
        'display': 'Total Yards (Rush + Rec)'
    },
    'total_touchdowns': {
        'metrics': ['receiving_tds', 'rushing_tds'],
        'display': 'Total TDs (Rush + Rec)'
    },
    'overall_contribution': {
        'metrics': ['receiving_yards', 'rushing_yards', 'receiving_tds', 'rushing_tds', 'receptions', 'targets', 'carries'],
        'display': 'Overall Offensive Contribution',
        'weights': {  # Weights based on EPA and WPA analysis
            'receiving_yards': 1.0,      # Base unit for comparison
            'rushing_yards': 1.0,        # Equal to receiving yards in value
            'receiving_tds': 50.0,       # Based on EPA conversion (~7 points / 0.14 points per yard)
            'rushing_tds': 50.0,         # Equal to receiving TD in value
            'receptions': 8.0,           # Success rate and chain-moving value beyond yards
            'targets': 3.0,              # Opportunity cost and defensive attention value
            'carries': 3.0               # Similar opportunity value to targets
        }
    },
    'qb_contribution': {
        'metrics': ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'completions', 'attempts'],
        'display': 'QB Overall Contribution',
        'weights': {
            'passing_yards': 1.0,        # Base unit
            'passing_tds': 50.0,         # ~7 points / 0.14 per yard
            'rushing_yards': 1.2,        # Slightly higher value for QB rushing
            'rushing_tds': 50.0,         # Equal TD value
            'completions': 5.0,          # Chain-moving and success rate value
            'attempts': -1.0             # Penalty for inefficiency (balanced by completions)
        }
    }
}

SKILL_POSITIONS = ['WR', 'RB', 'TE']


def is_season_complete(year: int) -> bool:
    """
    Check if a season is complete based on current date.

    Args:
        year: Season year to check

    Returns:
        True if season is complete, False if ongoing
    """
    from datetime import datetime

    current_year = datetime.now().year
    current_month = datetime.now().month

    # Previous years are always complete
    if year < current_year:
        return True

    # Current year: complete if February or later (after Super Bowl)
    if year == current_year and current_month >= 2:
        return True

    return False


def classify_player_profile(boom_count: int, steady_count: int, bust_count: int) -> str:
    """
    Classify player's overall performance profile based on game distribution.

    Args:
        boom_count: Number of boom games (>0.5 SD above mean)
        steady_count: Number of steady games (within ±0.5 SD of mean)
        bust_count: Number of bust games (<0.5 SD below mean)

    Returns:
        Profile classification string
    """
    total_games = boom_count + steady_count + bust_count

    if total_games == 0:
        return "N/A"

    steady_pct = steady_count / total_games
    volatile_pct = (boom_count + bust_count) / total_games

    # Classify based on distribution patterns
    if steady_pct >= 0.50:
        return "Consistent"
    elif volatile_pct >= 0.75:
        return "Boom/Bust"
    elif boom_count > bust_count * 1.5:
        return "High-Ceiling"
    elif bust_count > boom_count * 1.5:
        return "Low-Floor"
    else:
        return "Volatile"


def load_team_weekly_stats(year: int) -> pl.DataFrame:
    """Load team weekly stats for a given year, handling team relocations."""
    try:
        # Get all team data from the cache
        team_data = []
        team_cache_dir = Path(CACHE_DIR) / "team_stats"
        loaded_teams = set()  # Track which teams we've loaded to avoid duplicates

        # For each team directory, try to load data for this year
        for team_dir in team_cache_dir.iterdir():
            if team_dir.is_dir():
                team_code = team_dir.name

                # Skip if we've already loaded this team (handles relocation cases)
                if team_code in loaded_teams:
                    continue

                file_path = team_dir / f"{team_code}-{year}.csv"
                if file_path.exists():
                    try:
                        df = pl.read_csv(file_path, infer_schema_length=10000)

                        # Normalize team codes in the dataframe
                        df = normalize_team_codes_in_dataframe(df, year)

                        team_data.append(df)
                        loaded_teams.add(team_code)

                        # For relocated teams, mark the other code as loaded too
                        for old_code, info in TEAM_RELOCATIONS.items():
                            if team_code == old_code:
                                loaded_teams.add(info["new_code"])
                            elif team_code == info["new_code"]:
                                loaded_teams.add(old_code)

                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {str(e)}")
                        continue
        
        if not team_data:
            logger.error(f"No team data found for {year}")
            return None
        
        # Align schemas before concatenating
        if not team_data:
            logger.error(f"No team data found for {year}")
            return None
        
        # Cast all dataframes to have consistent schema
        try:
            # Get union of all columns
            all_cols = set()
            for df in team_data:
                all_cols.update(df.columns)
            
            # For each dataframe, add missing columns as nulls and cast to strings where needed
            aligned_data = []
            for df in team_data:
                # Add missing columns as null
                for col in all_cols:
                    if col not in df.columns:
                        df = df.with_columns(pl.lit(None).alias(col))
                
                # Cast all columns to string temporarily for concat
                schema = {col: pl.Utf8 for col in df.columns}
                df = df.cast(schema, strict=False)
                aligned_data.append(df)
            
            # Concat all dataframes
            combined_df = pl.concat(aligned_data, how="vertical")
            
            # Cast numeric columns back
            numeric_cols = ['week', 'completions', 'attempts', 'passing_yards', 'passing_tds', 
                          'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds',
                          'receptions', 'targets', 'carries', 'sacks', 'interceptions',
                          'def_tds', 'special_teams_tds', 'fg_made', 'pat_made']
            for col in numeric_cols:
                if col in combined_df.columns:
                    combined_df = combined_df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
            
            return combined_df
        except Exception as e:
            logger.error(f"Error concatenating team data for {year}: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error loading team stats for {year}: {str(e)}")
        return None

def load_position_weekly_stats(year: int, position: str) -> pl.DataFrame:
    """Load weekly stats for a specific position and year."""
    try:
        file_path = Path(CACHE_DIR) / "positional_player_stats" / position.lower() / f"{position.lower()}-{year}.csv"
        if not file_path.exists():
            logger.error(f"No {position} data found for {year}")
            return None

        df = pl.read_csv(file_path)

        # Convert problematic list columns to String type
        for col in ['fg_blocked_list', 'fg_missed_list', 'fg_made_list',
                   'fg_made_distance', 'fg_missed_distance', 'fg_blocked_distance', 'gwfg_distance']:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False))

        # Convert columns that should be Float64 but might be String
        float_columns = ['pacr', 'passing_cpoe', 'passing_epa', 'racr', 'rushing_epa', 'receiving_epa']
        for col in float_columns:
            if col in df.columns and df[col].dtype == pl.Utf8:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        # Normalize team codes to handle relocations
        df = normalize_team_codes_in_dataframe(df, year)

        return df
    except Exception as e:
        logger.error(f"Error loading {position} stats for {year}: {str(e)}")
        return None

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

def generate_weekly_tables(year: int) -> str:
    """Generate weekly offensive share tables for all teams."""
    logger.info(f"Generating weekly offensive share tables for {year}")
    
    # Load team stats
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team data available."
        
    # Load player stats for each position
    position_stats = {}
    for pos in SKILL_POSITIONS:
        df = load_position_weekly_stats(year, pos)
        if df is not None:
            position_stats[pos] = df
    
    if not position_stats:
        return "No player data available."
    
    # Combine all position stats
    player_stats = pl.concat(list(position_stats.values()))
    
    # Generate tables for each week
    markdown = f"# Offensive Share Analysis - {year}\n\n"
    
    # Get unique teams
    teams = sorted(team_stats['team'].unique().to_list())
    weeks = sorted(team_stats['week'].unique().to_list())
    
    for team in teams:
        markdown += f"## {team}\n\n"
        
        for week in weeks:
            markdown += f"### Week {week}\n\n"
            
            # Create a table for each skill position metric
            for metric, metric_name in SKILL_POSITION_METRICS.items():
                try:
                    shares = calculate_offensive_shares(
                        team_stats.filter(pl.col('team') == team).filter(pl.col('week') == week),
                        player_stats.filter(pl.col('team') == team).filter(pl.col('week') == week),
                        metric,
                        opponent_stats=team_stats,
                        year=year,
                        week=week,
                        team=team
                    )
                    
                    if shares.height == 0:
                        continue
                    
                    table = PrettyTable()
                    table.title = f"{metric_name} Share"
                    table.field_names = ["Player", "Position", "Raw Share (%)", "Adjusted Share (%)"]
                    table.align = "l"
                    table.float_format = '.1'
                    table.set_style(TableStyle.MARKDOWN)
                    
                    # Get top 5 contributors by adjusted share
                    adj_metric = f"{metric}_adjusted" if f"{metric}_adjusted" in shares.columns else f"{metric}"
                    top_players = shares.sort(f'{adj_metric}_share', descending=True).head(5)
                    
                    for row in top_players.iter_rows(named=True):
                        player_name = row['player_name']
                        position = player_stats.filter(pl.col('player_name') == player_name)['position'].head(1)[0]
                        raw_share = row[f'{metric}_share']
                        adj_share = row[f'{adj_metric}_share']
                        table.add_row([player_name, position, raw_share, adj_share])
                    
                    return table.get_string()
                    
                    markdown += "\n"
                except Exception as e:
                    logger.error(f"Error generating table for {team} week {week} {metric}: {str(e)}")
                    continue
                    
        markdown += "---\n\n"
    
    return markdown

def generate_season_summary(year: int) -> str:
    """Generate season summary table showing top offensive share holder per team."""
    logger.info(f"Generating season summary for {year}")
    
    # Load team stats
    logger.info("Loading team stats...")
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team data available."
    logger.info(f"Loaded {team_stats.height} team-week records")
        
    # Load player stats for each position
    logger.info("Loading player stats...")
    position_stats = {}
    for pos in SKILL_POSITIONS:
        df = load_position_weekly_stats(year, pos)
        if df is not None:
            position_stats[pos] = df
    
    if not position_stats:
        return "No player data available."
    logger.info(f"Loaded player stats for {len(position_stats)} positions")
    
    # Combine all position stats
    player_stats = pl.concat(list(position_stats.values()))
    logger.info(f"Combined {player_stats.height} player-week records")
    
    # Get unique teams
    teams = sorted(team_stats['team'].unique().to_list())
    logger.info(f"Processing {len(teams)} teams")
    
    markdown = f"# Season Summary - {year}\n\n"
    
    # Create a table for each skill position metric
    logger.info(f"Creating tables for {len(SKILL_POSITION_METRICS)} metrics")
    for metric, metric_name in SKILL_POSITION_METRICS.items():
        logger.info(f"Processing metric: {metric_name}")
            
        table = PrettyTable()
        table.title = f"{metric_name} Share Leaders"
        table.field_names = ["Team", "Player", "Position", "Season Share (%)"]
        table.align = "l"
        table.float_format = '.1'
        table.set_style(TableStyle.MARKDOWN)
        
        logger.info(f"Processing {len(teams)} teams for {metric_name}")
        for team_idx, team in enumerate(teams):
            if team_idx % 5 == 0:
                logger.info(f"  [{metric_name}] Processing team {team_idx+1}/{len(teams)}: {team}")
            try:
                # Get all weeks for this team
                team_season = team_stats.filter(pl.col('team') == team)
                player_season = player_stats.filter(pl.col('team') == team)
                
                # Calculate week-by-week shares with situational adjustments
                all_shares = []
                for week in team_season['week'].unique().to_list():
                    team_week = team_season.filter(pl.col('week') == week)
                    player_week = player_season.filter(pl.col('week') == week)
                    
                    week_shares = calculate_offensive_shares(team_week, player_week, metric, opponent_stats=team_stats, year=year, week=week, team=team)
                    if week_shares.height > 0:
                        all_shares.append(week_shares)
                
                if not all_shares:
                    continue
                
                # Combine all weeks
                combined_shares = pl.concat(all_shares)
                
                # Average the share percentages across all weeks
                player_avg_shares = (
                    combined_shares
                    .group_by('player_name')
                    .agg([
                        pl.col(f'{metric}_share').mean().alias('avg_share')
                    ])
                    .sort('avg_share', descending=True)
                )
                
                if player_avg_shares.height == 0:
                    continue
                
                # Get the top contributor
                top_player = player_avg_shares.head(1)
                player_name = top_player['player_name'][0]
                position = player_stats.filter(pl.col('player_name') == player_name)['position'].head(1)[0]
                share = top_player['avg_share'][0]
                
                table.add_row([team, player_name, position, share])
            except Exception as e:
                logger.error(f"Error generating summary for {team} {metric}: {str(e)}")
                continue
        
        markdown += table.get_string() + "\n\n"
    
    return markdown


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
            # Ensure play_id is Int64 (handles both Float64 and Int64 sources)
            if 'play_id' in pbp_data.columns:
                pbp_data = pbp_data.with_columns([
                    pl.col('play_id').cast(pl.Int64, strict=False)
                ])
            # Polars coerces Int64 to Int32 automatically during join
            # Join FTN data with PBP
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


def generate_qb_rankings(year: int) -> str:
    """Generate a separate QB rankings table using full adjustment pipeline."""
    logger.info(f"Generating QB rankings for {year}")
    
    # Load QB stats
    qb_stats = load_position_weekly_stats(year, 'QB')
    if qb_stats is None:
        return "No QB data available."
    
    # Load team stats for opponent defensive adjustments
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team stats available for opponent adjustments."
    
    # Apply minimum activity threshold: 14 pass attempts per game (1978-now standard)
    # Count only games with meaningful participation (10+ attempts)
    # First, filter to only include games with 10+ attempts
    qb_stats = qb_stats.filter(pl.col('attempts') >= 10)
    
    qb_games = qb_stats.group_by(['player_id', 'player_name']).agg([
        pl.col('week').count().alias('games'),
        pl.col('attempts').sum().alias('total_attempts')
    ])
    
    qb_games = qb_games.with_columns([
        (pl.col('total_attempts') / pl.col('games')).alias('attempts_per_game')
    ])
    
    # Filter to qualified QBs (14+ attempts per game)
    qualified_qbs = qb_games.filter(pl.col('attempts_per_game') >= 14.0)
    qb_stats = qb_stats.join(
        qualified_qbs.select(['player_id', 'player_name']), 
        on=['player_id', 'player_name'], 
        how='inner'
    )
    
    if len(qb_stats) == 0:
        return "No qualified QBs (14+ attempts/game) for this season."
    
    logger.info(f"Filtered to {len(qualified_qbs)} qualified QBs (14+ attempts/game)")
    

    # Calculate QB contribution scores using play-by-play data with contextual penalties/rewards
    # This gives ABSOLUTE production (not team shares) - correct scale for comparing QBs across teams
    qb_contributions = []

    # Check if FTN data is available for this year
    from modules.ftn_cache_builder import FTN_START_YEAR
    has_ftn = year >= FTN_START_YEAR

    if has_ftn:
        logger.info(f"Calculating QB contributions with FTN contextual adjustments (PA, OOP, blitz, screen)...")
    else:
        logger.info(f"Calculating QB contributions from play-by-play data with contextual penalties and pressure bonuses...")

    for qb_name in qb_stats['player_name'].unique().to_list():
        qb_data = qb_stats.filter(pl.col('player_name') == qb_name)

        # Calculate contribution from PBP data (includes penalties and pressure bonuses)
        contribution = calculate_qb_contribution_from_pbp(year, qb_name)

        games_played = qb_data.height
        if games_played > 0:
            player_id = qb_data['player_id'][0]
            team = qb_data['recent_team'][0] if 'recent_team' in qb_data.columns else qb_data['team'][0]

            qb_contributions.append({
                'player_id': player_id,
                'player_name': qb_name,
                'team': team,
                'contribution': contribution,
                'games': games_played,
                'avg_per_game': contribution / games_played
            })

    if not qb_contributions:
        return "No QB contributions calculated."

    # Create DataFrame with raw scores
    qb_season = pl.DataFrame(qb_contributions).rename({'contribution': 'raw_score'})

    # Calculate average opponent defense multipliers for each QB
    # Apply opponent defense metrics to QB weekly stats
    qb_weekly_with_defense = adjust_for_opponent(qb_stats, team_stats)

    # Calculate average pass_defense_multiplier per QB
    avg_opponent_defense = qb_weekly_with_defense.group_by(['player_id', 'player_name']).agg([
        pl.col('pass_defense_multiplier').mean().alias('avg_opponent_defense_mult')
    ])

    # Join opponent defense multiplier with season totals
    qb_season = qb_season.join(
        avg_opponent_defense.select(['player_id', 'avg_opponent_defense_mult']),
        on='player_id',
        how='left'
    ).with_columns([
        # Fill missing with 1.0 and clip to reasonable range
        pl.col('avg_opponent_defense_mult').fill_null(1.0).clip(0.5, 2.5).alias('avg_opponent_defense_mult')
    ])

    # Apply opponent defense adjustment to raw score
    qb_season = qb_season.with_columns([
        (pl.col('raw_score') * pl.col('avg_opponent_defense_mult')).alias('defense_adjusted_score')
    ])

    # Calculate average difficulty multipliers
    avg_difficulty = calculate_average_difficulty(year, qb_stats)
    if avg_difficulty is not None:
        qb_season = qb_season.join(
            avg_difficulty.select(['player_id', 'avg_difficulty_multiplier']),
            on='player_id',
            how='left'
        )
    
    # Calculate team-level max games (to penalize QBs who missed games their team played)
    team_games = qb_stats.group_by('team').agg([
        pl.col('week').n_unique().alias('team_max_games')
    ])
    
    # Join team max games
    qb_season = qb_season.join(team_games, on='team', how='left')
    
    # Calculate hybrid score: defense_adjusted_score * (qb_games / team_games)^0.4
    # This penalizes QBs who missed games their team played, but not for bye weeks
    qb_season = qb_season.with_columns([
        (pl.col('defense_adjusted_score') * (pl.col('games') / pl.col('team_max_games')).pow(0.4)).alias('hybrid_score')
    ])
    
    # Z-score normalization for hybrid scores
    mean_hybrid = qb_season['hybrid_score'].mean()
    std_hybrid = qb_season['hybrid_score'].std()
    
    qb_season = qb_season.with_columns([
        (50 + ((pl.col('hybrid_score') - mean_hybrid) / std_hybrid) * 17.5).alias('normalized_hybrid')
    ])
    
    # Sort by normalized hybrid score
    qb_season = qb_season.sort('normalized_hybrid', descending=True)
    
    # Create table
    markdown = f"# QB Rankings - {year}\n\n"
    table = PrettyTable()
    table.field_names = ["Rank", "QB", "Team", "Games", "Raw", "Def Adj", "Opp Def", "Difficulty", "Avg/Game", "Normalized"]
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)

    for rank, row in enumerate(qb_season.iter_rows(named=True), 1):
        games = row['games']
        raw = row['raw_score']
        defense_adjusted = row['defense_adjusted_score']
        opponent_defense_mult = row.get('avg_opponent_defense_mult', 1.0)
        avg_per_game = row['avg_per_game']
        difficulty = row.get('avg_difficulty_multiplier', 1.0)
        normalized = row['normalized_hybrid']

        table.add_row([
            rank,
            row['player_name'],
            row['team'].upper(),
            games,
            f"{raw:.2f}",
            f"{defense_adjusted:.2f}",
            f"{opponent_defense_mult:.3f}",
            f"{difficulty:.3f}",
            f"{avg_per_game:.2f}",
            f"{normalized:.2f}"
        ])
    
    markdown += table.get_string() + "\n\n"
    return markdown


def generate_rb_rankings(year: int) -> str:
    """Generate comprehensive RB rankings showing all qualified players using full adjustment pipeline."""
    logger.info(f"Generating RB rankings for {year}")
    
    # Load team stats for opponent adjustments
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team data available."
    
    # Load RB stats
    rb_stats = load_position_weekly_stats(year, 'RB')
    if rb_stats is None:
        return "No RB data available."
    
    # Apply minimum activity threshold: 6.25 carries per game (matching top_contributors)
    rb_games = rb_stats.group_by(['player_id', 'player_name']).agg([
        pl.col('week').count().alias('games'),
        pl.col('carries').sum().alias('total_carries')
    ])
    
    rb_games = rb_games.with_columns([
        (pl.col('total_carries') / pl.col('games')).alias('carries_per_game')
    ])
    
    # Filter to qualified RBs (6.25+ carries per game)
    qualified_rbs = rb_games.filter(pl.col('carries_per_game') >= 6.25)
    rb_stats = rb_stats.join(
        qualified_rbs.select(['player_id', 'player_name']), 
        on=['player_id', 'player_name'], 
        how='inner'
    )
    
    if len(rb_stats) == 0:
        return "No qualified RBs (6.25+ carries/game) for this season."
    
    logger.info(f"Filtered to {len(qualified_rbs)} qualified RBs (6.25+ carries/game)")
    
    # Use the full calculate_offensive_shares pipeline with defensive adjustments
    # This includes: game situation adjustments, opponent rolling averages, all multipliers
    raw_contributions = calculate_offensive_shares(
        team_stats, 
        rb_stats, 
        'overall_contribution',
        opponent_stats=team_stats,
        year=None  # year=None means no situational adjustments for "raw" score
    )
    
    adjusted_contributions = calculate_offensive_shares(
        team_stats, 
        rb_stats, 
        'overall_contribution',
        opponent_stats=team_stats,
        year=year  # year=year enables situational adjustments
    )
    
    # Add position column back for Phase 4 adjustments
    adjusted_contributions = adjusted_contributions.with_columns([
        pl.lit('RB').alias('position')
    ])

    # Apply Phase 4 adjustments (catch rate + blocking quality)
    adjusted_contributions = apply_phase4_adjustments(adjusted_contributions, year)

    # Apply Phase 4.5 adjustments (weather-based performance)
    adjusted_contributions = apply_phase4_5_weather_adjustments(adjusted_contributions, year, 'RB')

    # Save per-game contributions before Phase 5 for consistency metrics
    per_game_contributions = adjusted_contributions.clone().select([
        'player_id', 'player_name', 'team', 'week', 'player_overall_contribution'
    ])
    
    # Apply Phase 5 adjustments (talent context + sample size dampening)
    adjusted_contributions = apply_phase5_adjustments(adjusted_contributions, year)
    
    # Aggregate to season totals
    # Raw scores (no adjustments)
    raw_season = raw_contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').sum().alias('raw_score'),
        pl.col('week').count().alias('games')
    ])
    
    # Adjusted scores (with Phase 4 & 5)
    # After Phase 5, all weeks have the same dampened score, so we use mean() to get the per-game value
    adjusted_season = adjusted_contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').mean().alias('adjusted_score')
    ])
    
    # Join raw and adjusted
    rb_season = raw_season.join(adjusted_season, on=['player_id', 'player_name', 'team'], how='left')
    
    # Calculate average difficulty multiplier
    avg_difficulty = calculate_average_difficulty(year, rb_stats)
    if avg_difficulty is not None:
        rb_season = rb_season.join(
            avg_difficulty.select(['player_id', 'player_name', 'avg_difficulty_multiplier']),
            on=['player_id', 'player_name'],
            how='left'
        )
    else:
        rb_season = rb_season.with_columns([
            pl.lit(1.0).alias('avg_difficulty_multiplier')
        ])
    
    # Calculate consistency metrics from per_game_contributions (pre-Phase 5)
    # We need to scale these by the Phase 5 factor to match the adjusted scores
    
    # Get pre-Phase 5 means
    pre_phase5_means = (
        per_game_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('pre_phase5_mean')
        ])
    )
    
    # Get post-Phase 5 values (all weeks have same dampened value after Phase 5)
    post_phase5_values = (
        adjusted_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('post_phase5_value')
        ])
    )
    
    # Calculate scaling factor
    scaling_factors = pre_phase5_means.join(post_phase5_values, on=['player_id', 'player_name', 'team'])
    scaling_factors = scaling_factors.with_columns([
        (pl.col('post_phase5_value') / pl.col('pre_phase5_mean')).alias('phase5_scaling_factor')
    ])
    
    # Calculate consistency metrics for each RB
    consistency_data = []
    for player_id in rb_season['player_id'].to_list():
        player_weeks = per_game_contributions.filter(pl.col('player_id') == player_id)
        if len(player_weeks) == 0:
            continue
            
        contributions = player_weeks['player_overall_contribution'].to_list()
        contributions.sort()
        
        # Get Phase 5 scaling factor for this player
        player_scaling = scaling_factors.filter(pl.col('player_id') == player_id)
        if len(player_scaling) > 0:
            scale_factor = player_scaling['phase5_scaling_factor'][0]
        else:
            scale_factor = 1.0
        
        # Calculate typical (25th/75th percentile average) and apply Phase 5 scaling
        if len(contributions) >= 4:
            q1_idx = len(contributions) // 4
            q3_idx = 3 * len(contributions) // 4
            typical = (contributions[q1_idx] + contributions[q3_idx]) / 2 * scale_factor
        else:
            typical = sum(contributions) / len(contributions) * scale_factor if contributions else 0
        
        # Calculate consistency using standard deviation-based thresholds
        # Use UNSCALED values for consistency counts
        unscaled_contribs = contributions  # Already unscaled
        mean_contrib = sum(unscaled_contribs) / len(unscaled_contribs)
        variance = sum((x - mean_contrib) ** 2 for x in unscaled_contribs) / len(unscaled_contribs)
        std_dev = variance ** 0.5

        # Define thresholds: ±0.5 SD from mean
        bust_threshold = mean_contrib - (0.5 * std_dev)
        boom_threshold = mean_contrib + (0.5 * std_dev)

        below_avg = sum(1 for g in unscaled_contribs if g < bust_threshold)
        at_avg = sum(1 for g in unscaled_contribs if bust_threshold <= g <= boom_threshold)
        above_avg = sum(1 for g in unscaled_contribs if g > boom_threshold)
        
        # Calculate trend/profile based on season completion status
        max_week_in_season = rb_stats['week'].max()
        last_week_played = player_weeks['week'].max()
        weeks_since_last_game = max_week_in_season - last_week_played

        if is_season_complete(year):
            # Show performance profile for completed seasons (even if player finished injured)
            trend_str = classify_player_profile(above_avg, at_avg, below_avg)
        elif weeks_since_last_game > 2:
            trend_str = "INACTIVE"
        elif len(contributions) >= 6:
            # Show percentage trend for ongoing seasons
            midpoint = len(contributions) // 2
            first_half = sum(contributions[:midpoint]) / midpoint
            second_half = sum(contributions[midpoint:]) / (len(contributions) - midpoint)
            trend = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
            if abs(trend) < 5:
                trend_str = "Stable"
            else:
                trend_str = f"{trend:+.1f}%"
        else:
            trend_str = "N/A"
        
        player_name = player_weeks['player_name'][0]
        consistency_data.append({
            'player_id': player_id,
            'player_name': player_name,
            'typical': typical,
            'below_avg': below_avg,
            'at_avg': at_avg,
            'above_avg': above_avg,
            'trend': trend_str
        })
    
    if consistency_data:
        consistency_df = pl.DataFrame(consistency_data)
        rb_season = rb_season.join(consistency_df, on=['player_id', 'player_name'], how='left')
    
    # Calculate season total for sorting (adjusted per-game * games)
    rb_season = rb_season.with_columns([
        (pl.col('adjusted_score') * pl.col('games')).alias('adjusted_total')
    ])
    
    # Sort by adjusted season total
    rb_season = rb_season.sort('adjusted_total', descending=True)
    
    # Create table
    markdown = f"# RB Rankings - {year}\n\n"
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Games", "Raw", "Adjusted", "Difficulty", "Avg/Game", "Typical", "Consistency", "Trend"]
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)
    
    for rank, row in enumerate(rb_season.iter_rows(named=True), 1):
        games = row['games']
        raw = row['raw_score']
        adjusted_per_game = row['adjusted_score']  # Phase 5 returns per-game dampened score
        adjusted_total = adjusted_per_game * games  # Calculate season total for display
        difficulty = row.get('avg_difficulty_multiplier', 1.0)
        # Handle None difficulty
        if difficulty is None:
            difficulty = 1.0
        avg_per_game = adjusted_per_game  # Already per-game from Phase 5
        typical = row.get('typical', avg_per_game)
        # Handle None typical
        if typical is None:
            typical = avg_per_game
        below = row.get('below_avg', 0)
        at = row.get('at_avg', 0)
        above = row.get('above_avg', 0)
        consistency = f"{below}/{at}/{above}"
        trend = row.get('trend', 'N/A')

        table.add_row([
            rank,
            row['player_name'],
            row['team'].upper(),
            games,
            f"{raw:.2f}",
            f"{adjusted_total:.2f}",
            f"{difficulty:.3f}",
            f"{avg_per_game:.2f}",
            f"{typical:.2f}",
            consistency,
            trend
        ])

    markdown += table.get_string() + "\n\n"
    return markdown


def generate_wr_rankings(year: int) -> str:
    """Generate comprehensive WR rankings showing all qualified players."""
    logger.info(f"Generating WR rankings for {year}")
    
    # Load WR stats
    wr_stats = load_position_weekly_stats(year, 'WR')
    if wr_stats is None:
        return "No WR data available."
    
    # Load team stats for opponent defensive adjustments
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team stats available for opponent adjustments."
    
    # Apply minimum activity threshold: 3 targets per game
    wr_games = wr_stats.group_by(['player_id', 'player_name']).agg([
        pl.col('week').count().alias('games'),
        pl.col('targets').sum().alias('total_targets')
    ])
    
    wr_games = wr_games.with_columns([
        (pl.col('total_targets') / pl.col('games')).alias('targets_per_game')
    ])
    
    # Filter to qualified WRs (3+ targets per game)
    qualified_wrs = wr_games.filter(pl.col('targets_per_game') >= 3.0)
    wr_stats = wr_stats.join(
        qualified_wrs.select(['player_id', 'player_name']), 
        on=['player_id', 'player_name'], 
        how='inner'
    )
    
    if len(wr_stats) == 0:
        return "No qualified WRs (3+ targets/game) for this season."
    
    logger.info(f"Filtered to {len(qualified_wrs)} qualified WRs (3+ targets/game)")
    
    # Use the full pipeline for WR contributions (receiving only)
    # Calculate raw contributions (no year context for baseline)
    raw_contributions = calculate_offensive_shares(
        team_stats, 
        wr_stats, 
        'overall_contribution',
        opponent_stats=team_stats,  # Include opponent defensive adjustments
        year=None  # No year context for raw baseline
    )
    
    # Calculate adjusted contributions (with all context including opponent defense)
    adjusted_contributions = calculate_offensive_shares(
        team_stats,
        wr_stats,
        'overall_contribution', 
        opponent_stats=team_stats,  # Include opponent defensive adjustments
        year=year  # Include year context for situational adjustments
    )
    
    # Add position column back for Phase 4 adjustments
    adjusted_contributions = adjusted_contributions.with_columns([
        pl.lit('WR').alias('position')
    ])

    # Apply Phase 4 adjustments (catch rate + blocking quality)
    adjusted_contributions = apply_phase4_adjustments(adjusted_contributions, year)

    # Apply Phase 4.5 adjustments (weather-based performance)
    adjusted_contributions = apply_phase4_5_weather_adjustments(adjusted_contributions, year, 'WR')

    # Save per-game contributions before Phase 5 for consistency metrics
    per_game_contributions = adjusted_contributions.clone().select([
        'player_id', 'player_name', 'team', 'week', 'player_overall_contribution'
    ])
    
    # Apply Phase 5 adjustments (talent context + sample size dampening)
    adjusted_contributions = apply_phase5_adjustments(adjusted_contributions, year)
    
    # Aggregate to season totals
    raw_season = raw_contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('week').count().alias('games'),
        pl.col('player_overall_contribution').sum().alias('raw_score')
    ])
    
    # After Phase 5, all weeks have the same dampened score, so use mean() to get per-game value
    adjusted_season = adjusted_contributions.group_by(['player_id', 'player_name']).agg([
        pl.col('player_overall_contribution').mean().alias('adjusted_score')
    ])
    
    # Join raw and adjusted scores
    wr_season = raw_season.join(adjusted_season, on=['player_id', 'player_name'], how='left')
    
    # Calculate average difficulty multipliers
    avg_difficulty = calculate_average_difficulty(year, wr_stats)
    if avg_difficulty is not None:
        wr_season = wr_season.join(
            avg_difficulty.select(['player_id', 'avg_difficulty_multiplier']),
            on='player_id',
            how='left'
        )
    
    # Calculate consistency metrics from per_game_contributions (pre-Phase 5)
    # We need to scale these by the Phase 5 factor to match the adjusted scores
    
    # Get pre-Phase 5 means
    pre_phase5_means = (
        per_game_contributions.group_by(['player_id', 'player_name'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('pre_phase5_mean')
        ])
    )
    
    # Get post-Phase 5 values (all weeks have same dampened value after Phase 5)
    post_phase5_values = (
        adjusted_contributions.group_by(['player_id', 'player_name'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('post_phase5_value')
        ])
    )
    
    # Calculate scaling factor
    scaling_factors = pre_phase5_means.join(post_phase5_values, on=['player_id', 'player_name'])
    scaling_factors = scaling_factors.with_columns([
        (pl.col('post_phase5_value') / pl.col('pre_phase5_mean')).alias('phase5_scaling_factor')
    ])
    
    # Calculate consistency metrics from per-week contributions
    consistency_data = []
    for player_id in wr_season['player_id'].unique().to_list():
        player_weeks = per_game_contributions.filter(pl.col('player_id') == player_id)
        player_name = player_weeks['player_name'][0]
        
        game_contributions = player_weeks['player_overall_contribution'].to_list()
        game_contributions.sort()
        
        # Get Phase 5 scaling factor for this player
        player_scaling = scaling_factors.filter(pl.col('player_id') == player_id)
        if len(player_scaling) > 0:
            scale_factor = player_scaling['phase5_scaling_factor'][0]
        else:
            scale_factor = 1.0
        
        # Calculate typical and apply Phase 5 scaling
        if len(game_contributions) >= 4:
            q1_idx = len(game_contributions) // 4
            q3_idx = 3 * len(game_contributions) // 4
            typical = (game_contributions[q1_idx] + game_contributions[q3_idx]) / 2 * scale_factor
        else:
            typical = sum(game_contributions) / len(game_contributions) * scale_factor if game_contributions else 0
        
        # Calculate consistency using standard deviation-based thresholds
        # Use UNSCALED values for consistency counts
        unscaled_contribs = game_contributions  # Already unscaled
        mean_contrib = sum(unscaled_contribs) / len(unscaled_contribs)
        variance = sum((x - mean_contrib) ** 2 for x in unscaled_contribs) / len(unscaled_contribs)
        std_dev = variance ** 0.5

        # Define thresholds: ±0.5 SD from mean
        bust_threshold = mean_contrib - (0.5 * std_dev)
        boom_threshold = mean_contrib + (0.5 * std_dev)

        below_avg = sum(1 for g in unscaled_contribs if g < bust_threshold)
        at_avg = sum(1 for g in unscaled_contribs if bust_threshold <= g <= boom_threshold)
        above_avg = sum(1 for g in unscaled_contribs if g > boom_threshold)

        # Calculate trend/profile based on season completion status
        max_week_in_season = wr_stats['week'].max()
        last_week_played = player_weeks['week'].max()
        weeks_since_last_game = max_week_in_season - last_week_played

        if is_season_complete(year):
            # Show performance profile for completed seasons (even if player finished injured)
            trend_str = classify_player_profile(above_avg, at_avg, below_avg)
        elif weeks_since_last_game > 2:
            trend_str = "INACTIVE"
        elif len(game_contributions) >= 6:
            # Show percentage trend for ongoing seasons
            midpoint = len(game_contributions) // 2
            first_half = sum(game_contributions[:midpoint]) / midpoint
            second_half = sum(game_contributions[midpoint:]) / (len(game_contributions) - midpoint)
            trend = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
            if abs(trend) < 5:
                trend_str = "Stable"
            else:
                trend_str = f"{trend:+.1f}%"
        else:
            trend_str = "N/A"
        
        consistency_data.append({
            'player_id': player_id,
            'player_name': player_name,
            'typical': typical,
            'below_avg': below_avg,
            'at_avg': at_avg,
            'above_avg': above_avg,
            'trend': trend_str
        })
    
    # Join consistency data with season totals
    if consistency_data:
        consistency_df = pl.DataFrame(consistency_data)
        wr_season = wr_season.join(consistency_df, on=['player_id', 'player_name'], how='left')
    
    # Calculate season total for sorting (adjusted per-game * games)
    wr_season = wr_season.with_columns([
        (pl.col('adjusted_score') * pl.col('games')).alias('adjusted_total')
    ])
    
    # Sort by adjusted season total
    wr_season = wr_season.sort('adjusted_total', descending=True)
    
    # Create table
    markdown = f"# WR Rankings - {year}\n\n"
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Games", "Raw", "Adjusted", "Difficulty", "Avg/Game", "Typical", "Consistency", "Trend"]
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)
    
    for rank, row in enumerate(wr_season.iter_rows(named=True), 1):
        games = row['games']
        raw = row['raw_score']
        adjusted_per_game = row['adjusted_score']  # Phase 5 returns per-game dampened score
        adjusted_total = adjusted_per_game * games  # Calculate season total for display
        difficulty = row.get('avg_difficulty_multiplier', 1.0)
        # Handle None difficulty
        if difficulty is None:
            difficulty = 1.0
        avg_per_game = adjusted_per_game  # Already per-game from Phase 5
        typical = row.get('typical', avg_per_game)
        # Handle None typical
        if typical is None:
            typical = avg_per_game
        below = row.get('below_avg', 0)
        at = row.get('at_avg', 0)
        above = row.get('above_avg', 0)
        consistency = f"{below}/{at}/{above}"
        trend = row.get('trend', 'N/A')

        table.add_row([
            rank,
            row['player_name'],
            row['team'].upper(),
            games,
            f"{raw:.2f}",
            f"{adjusted_total:.2f}",
            f"{difficulty:.3f}",
            f"{avg_per_game:.2f}",
            f"{typical:.2f}",
            consistency,
            trend
        ])

    markdown += table.get_string() + "\n\n"
    return markdown


def generate_te_rankings(year: int) -> str:
    """Generate comprehensive TE rankings showing all qualified players."""
    logger.info(f"Generating TE rankings for {year}")
    
    # Load TE stats
    te_stats = load_position_weekly_stats(year, 'TE')
    if te_stats is None:
        return "No TE data available."
    
    # Load team stats for opponent defensive adjustments
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team stats available for opponent adjustments."
    
    # Apply minimum activity threshold: 2 targets per game
    te_games = te_stats.group_by(['player_id', 'player_name']).agg([
        pl.col('week').count().alias('games'),
        pl.col('targets').sum().alias('total_targets')
    ])
    
    te_games = te_games.with_columns([
        (pl.col('total_targets') / pl.col('games')).alias('targets_per_game')
    ])
    
    # Filter to qualified TEs (2+ targets per game)
    qualified_tes = te_games.filter(pl.col('targets_per_game') >= 2.0)
    te_stats = te_stats.join(
        qualified_tes.select(['player_id', 'player_name']), 
        on=['player_id', 'player_name'], 
        how='inner'
    )
    
    if len(te_stats) == 0:
        return "No qualified TEs (2+ targets/game) for this season."
    
    logger.info(f"Filtered to {len(qualified_tes)} qualified TEs (2+ targets/game)")
    
    # Use the full pipeline for TE contributions (receiving only)
    # Calculate raw contributions (no year context for baseline)
    raw_contributions = calculate_offensive_shares(
        team_stats, 
        te_stats, 
        'overall_contribution',
        opponent_stats=team_stats,  # Include opponent defensive adjustments
        year=None  # No year context for raw baseline
    )
    
    # Calculate adjusted contributions (with all context including opponent defense)
    adjusted_contributions = calculate_offensive_shares(
        team_stats,
        te_stats,
        'overall_contribution', 
        opponent_stats=team_stats,  # Include opponent defensive adjustments
        year=year  # Include year context for situational adjustments
    )
    
    # Add position column back for Phase 4 adjustments
    adjusted_contributions = adjusted_contributions.with_columns([
        pl.lit('TE').alias('position')
    ])

    # Apply Phase 4 adjustments (catch rate + blocking quality)
    adjusted_contributions = apply_phase4_adjustments(adjusted_contributions, year)

    # Apply Phase 4.5 adjustments (weather-based performance)
    adjusted_contributions = apply_phase4_5_weather_adjustments(adjusted_contributions, year, 'TE')

    # Save per-game contributions before Phase 5 for consistency metrics
    per_game_contributions = adjusted_contributions.clone().select([
        'player_id', 'player_name', 'team', 'week', 'player_overall_contribution'
    ])
    
    # Apply Phase 5 adjustments (talent context + sample size dampening)
    adjusted_contributions = apply_phase5_adjustments(adjusted_contributions, year)
    
    # Aggregate to season totals
    raw_season = raw_contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('week').count().alias('games'),
        pl.col('player_overall_contribution').sum().alias('raw_score')
    ])
    
    # After Phase 5, all weeks have the same dampened score, so use mean() to get per-game value
    adjusted_season = adjusted_contributions.group_by(['player_id', 'player_name']).agg([
        pl.col('player_overall_contribution').mean().alias('adjusted_score')
    ])
    
    # Join raw and adjusted scores
    te_season = raw_season.join(adjusted_season, on=['player_id', 'player_name'], how='left')
    
    # Calculate average difficulty multipliers
    avg_difficulty = calculate_average_difficulty(year, te_stats)
    if avg_difficulty is not None:
        te_season = te_season.join(
            avg_difficulty.select(['player_id', 'avg_difficulty_multiplier']),
            on='player_id',
            how='left'
        )
    
    # Calculate consistency metrics from per_game_contributions (pre-Phase 5)
    # We need to scale these by the Phase 5 factor to match the adjusted scores
    
    # Get pre-Phase 5 means
    pre_phase5_means = (
        per_game_contributions.group_by(['player_id', 'player_name'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('pre_phase5_mean')
        ])
    )
    
    # Get post-Phase 5 values (all weeks have same dampened value after Phase 5)
    post_phase5_values = (
        adjusted_contributions.group_by(['player_id', 'player_name'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('post_phase5_value')
        ])
    )
    
    # Calculate scaling factor
    scaling_factors = pre_phase5_means.join(post_phase5_values, on=['player_id', 'player_name'])
    scaling_factors = scaling_factors.with_columns([
        (pl.col('post_phase5_value') / pl.col('pre_phase5_mean')).alias('phase5_scaling_factor')
    ])
    
    # Calculate consistency metrics from per-week contributions
    consistency_data = []
    for player_id in te_season['player_id'].unique().to_list():
        player_weeks = per_game_contributions.filter(pl.col('player_id') == player_id)
        player_name = player_weeks['player_name'][0]
        
        game_contributions = player_weeks['player_overall_contribution'].to_list()
        game_contributions.sort()
        
        # Get Phase 5 scaling factor for this player
        player_scaling = scaling_factors.filter(pl.col('player_id') == player_id)
        if len(player_scaling) > 0:
            scale_factor = player_scaling['phase5_scaling_factor'][0]
        else:
            scale_factor = 1.0
        
        # Calculate typical and apply Phase 5 scaling
        if len(game_contributions) >= 4:
            q1_idx = len(game_contributions) // 4
            q3_idx = 3 * len(game_contributions) // 4
            typical = (game_contributions[q1_idx] + game_contributions[q3_idx]) / 2 * scale_factor
        else:
            typical = sum(game_contributions) / len(game_contributions) * scale_factor if game_contributions else 0
        
        # Calculate consistency using standard deviation-based thresholds
        # Use UNSCALED values for consistency counts
        unscaled_contribs = game_contributions  # Already unscaled
        mean_contrib = sum(unscaled_contribs) / len(unscaled_contribs)
        variance = sum((x - mean_contrib) ** 2 for x in unscaled_contribs) / len(unscaled_contribs)
        std_dev = variance ** 0.5

        # Define thresholds: ±0.5 SD from mean
        bust_threshold = mean_contrib - (0.5 * std_dev)
        boom_threshold = mean_contrib + (0.5 * std_dev)

        below_avg = sum(1 for g in unscaled_contribs if g < bust_threshold)
        at_avg = sum(1 for g in unscaled_contribs if bust_threshold <= g <= boom_threshold)
        above_avg = sum(1 for g in unscaled_contribs if g > boom_threshold)

        # Calculate trend/profile based on season completion status
        max_week_in_season = te_stats['week'].max()
        last_week_played = player_weeks['week'].max()
        weeks_since_last_game = max_week_in_season - last_week_played

        if is_season_complete(year):
            # Show performance profile for completed seasons (even if player finished injured)
            trend_str = classify_player_profile(above_avg, at_avg, below_avg)
        elif weeks_since_last_game > 2:
            trend_str = "INACTIVE"
        elif len(game_contributions) >= 6:
            # Show percentage trend for ongoing seasons
            midpoint = len(game_contributions) // 2
            first_half = sum(game_contributions[:midpoint]) / midpoint
            second_half = sum(game_contributions[midpoint:]) / (len(game_contributions) - midpoint)
            trend = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
            if abs(trend) < 5:
                trend_str = "Stable"
            else:
                trend_str = f"{trend:+.1f}%"
        else:
            trend_str = "N/A"
        
        consistency_data.append({
            'player_id': player_id,
            'player_name': player_name,
            'typical': typical,
            'below_avg': below_avg,
            'at_avg': at_avg,
            'above_avg': above_avg,
            'trend': trend_str
        })
    
    # Join consistency data with season totals
    if consistency_data:
        consistency_df = pl.DataFrame(consistency_data)
        te_season = te_season.join(consistency_df, on=['player_id', 'player_name'], how='left')
    
    # Calculate season total for sorting (adjusted per-game * games)
    te_season = te_season.with_columns([
        (pl.col('adjusted_score') * pl.col('games')).alias('adjusted_total')
    ])
    
    # Sort by adjusted season total
    te_season = te_season.sort('adjusted_total', descending=True)
    
    # Create table
    markdown = f"# TE Rankings - {year}\n\n"
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Games", "Raw", "Adjusted", "Difficulty", "Avg/Game", "Typical", "Consistency", "Trend"]
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)
    
    for rank, row in enumerate(te_season.iter_rows(named=True), 1):
        games = row['games']
        raw = row['raw_score']
        adjusted_per_game = row['adjusted_score']  # Phase 5 returns per-game dampened score
        adjusted_total = adjusted_per_game * games  # Calculate season total for display
        difficulty = row.get('avg_difficulty_multiplier', 1.0)
        # Handle None difficulty
        if difficulty is None:
            difficulty = 1.0
        avg_per_game = adjusted_per_game  # Already per-game from Phase 5
        typical = row.get('typical', avg_per_game)
        # Handle None typical
        if typical is None:
            typical = avg_per_game
        below = row.get('below_avg', 0)
        at = row.get('at_avg', 0)
        above = row.get('above_avg', 0)
        consistency = f"{below}/{at}/{above}"
        trend = row.get('trend', 'N/A')

        table.add_row([
            rank,
            row['player_name'],
            row['team'].upper(),
            games,
            f"{raw:.2f}",
            f"{adjusted_total:.2f}",
            f"{difficulty:.3f}",
            f"{avg_per_game:.2f}",
            f"{typical:.2f}",
            consistency,
            trend
        ])

    markdown += table.get_string() + "\n\n"
    return markdown


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

        # Debug: Check team name formats
        if len(contributions) > 0:
            sample_contrib_team = contributions['team'][0]
            sample_pbp_teams = pbp_data.select(['home_team', 'away_team']).head(1)
            logger.info(f"DEBUG - Sample contribution team: '{sample_contrib_team}' (type: {type(sample_contrib_team)})")
            logger.info(f"DEBUG - Sample PBP teams: {sample_pbp_teams}")

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


def generate_ftn_context(year: int, player_stats: pl.DataFrame, season_contributions: pl.DataFrame) -> str:
    """Generate FTN charting context tables for QB/RB/WR/TE.
    
    Only generates content for years 2022+ (when FTN data is available).
    
    Args:
        year: Season year
        player_stats: Weekly player stats with FTN flags
        season_contributions: Season-level contribution scores
        
    Returns:
        Markdown string with FTN context tables
    """
    from modules.ftn_cache_builder import FTN_START_YEAR
    
    if year < FTN_START_YEAR:
        return ""
    
    # Load FTN cache
    ftn_path = Path("cache/ftn") / f"ftn_{year}.parquet"
    if not ftn_path.exists():
        logger.warning(f"FTN cache not found for {year}, skipping FTN context")
        return ""
    
    try:
        ftn_data = pl.read_parquet(ftn_path)
    except Exception as e:
        logger.error(f"Error loading FTN cache for {year}: {e}")
        return ""
    
    # Load PBP data to join FTN with player stats
    pbp_path = Path("cache/pbp") / f"pbp_{year}.parquet"
    if not pbp_path.exists():
        logger.warning(f"PBP cache not found for {year}, cannot generate FTN context")
        return ""
    
    try:
        pbp_data = pl.read_parquet(pbp_path)
    except Exception as e:
        logger.error(f"Error loading PBP cache for {year}: {e}")
        return ""
    
    # Join FTN with PBP - handle column name variations and type casting
    # PBP cache may have 'game_id'/'play_id' or 'nflverse_game_id'/'nflverse_play_id'
    # Ensure types match for join
        # Ensure play_id is Int64 (handles both Float64 and Int64 sources)
        if 'play_id' in pbp_data.columns and 'nflverse_play_id' in pbp_data.columns:
            pbp_data = pbp_data.with_columns([
                pl.col('nflverse_play_id').cast(pl.Int64, strict=False)
            ])
        elif 'play_id' in pbp_data.columns:
            pbp_data = pbp_data.with_columns([
                pl.col('play_id').cast(pl.Int64, strict=False)
            ])
    try:
        # Rename PBP columns if needed to match FTN
        if 'game_id' in pbp_data.columns and 'nflverse_game_id' not in pbp_data.columns:
            pbp_data = pbp_data.rename({'game_id': 'nflverse_game_id'})
        if 'play_id' in pbp_data.columns and 'nflverse_play_id' not in pbp_data.columns:
            pbp_data = pbp_data.rename({'play_id': 'nflverse_play_id'})
        
        
        pbp_with_ftn = pbp_data.join(
            ftn_data,
            on=['nflverse_game_id', 'nflverse_play_id'],
            how='left'
        )
    except Exception as e:
        logger.error(f"Error joining FTN with PBP for {year}: {e}")
        return ""
    
    md = "## FTN Context (2022+)\n\n"
    md += "*Human-charted play characteristics from Football Technology Network*\n\n"
    
    # QB Play Style Context
    qb_players = player_stats.filter(pl.col('position') == 'QB').select(['player_id', 'player_name', 'position']).unique()
    if len(qb_players) > 0:
        qb_pbp = pbp_with_ftn.filter(
            (pl.col('passer_player_id').is_in(qb_players['player_id'].to_list())) &
            (pl.col('play_type') == 'pass') &
            (pl.col('is_play_action').is_not_null())  # Ensure FTN data exists
        )
        
        if len(qb_pbp) > 0:
            qb_ftn = qb_pbp.group_by('passer_player_id').agg([
                pl.len().alias('pass_attempts'),
                pl.col('is_play_action').mean().alias('pa_rate'),
                pl.col('is_qb_out_of_pocket').mean().alias('oop_rate'),
                pl.col('is_screen_pass').mean().alias('screen_rate'),
                (pl.col('n_blitzers') >= 5).mean().alias('blitz_rate')
            ]).rename({'passer_player_id': 'player_id'})
            
            # Join with player names and season scores
            qb_ftn = qb_ftn.join(qb_players, on='player_id', how='left')
            qb_season = season_contributions.filter(pl.col('position') == 'QB').head(15)
            qb_with_ftn = qb_season.join(qb_ftn, on=['player_id', 'player_name'], how='left')
            
            # Filter to QBs with at least 100 pass attempts
            qb_with_ftn = qb_with_ftn.filter(pl.col('pass_attempts') >= 100)
            
            if len(qb_with_ftn) > 0:
                md += "### QB - Play Style Context\n\n"
                md += "*How often QBs face different play types*\n\n"
                md += "- **Play Action**: -10% adjustment (easier)\n"
                md += "- **Out of Pocket**: +3 pts/completion (harder)\n"
                md += "- **5+ Blitzers**: -2 pts/completion (harder)\n"
                md += "- **Screen Pass**: -15% adjustment (easier)\n\n"
                
                qb_table = PrettyTable()
                qb_table.field_names = ["Player", "PA%", "OOP%", "Screen%", "Blitz%", "Pass Att", "Net Adjustment"]
                qb_table.align = "l"
                qb_table.float_format = '.1'
                qb_table.set_style(TableStyle.MARKDOWN)
                
                for row in qb_with_ftn.sort('raw_score', descending=True).iter_rows(named=True):
                    player_name = row['player_name']
                    pa_pct = row['pa_rate'] * 100 if row['pa_rate'] is not None else 0
                    oop_pct = row['oop_rate'] * 100 if row['oop_rate'] is not None else 0
                    screen_pct = row['screen_rate'] * 100 if row['screen_rate'] is not None else 0
                    blitz_pct = row['blitz_rate'] * 100 if row['blitz_rate'] is not None else 0
                    pass_att = int(row['pass_attempts']) if row['pass_attempts'] is not None else 0
                    
                    # Calculate net adjustment impact (simplified)
                    # Positive = harder, Negative = easier
                    net_adj = (oop_pct * 0.03) + (blitz_pct * -0.02) + (pa_pct * -0.10) + (screen_pct * -0.15)
                    adj_display = f"+{net_adj:.1f}%" if net_adj > 0 else f"{net_adj:.1f}%"
                    
                    qb_table.add_row([player_name, f"{pa_pct:.1f}%", f"{oop_pct:.1f}%", 
                                     f"{screen_pct:.1f}%", f"{blitz_pct:.1f}%", pass_att, adj_display])
                
                md += qb_table.get_string() + "\n\n"
    
    # RB Scheme Context
    rb_players = player_stats.filter(pl.col('position') == 'RB').select(['player_id', 'player_name', 'position']).unique()
    if len(rb_players) > 0:
        rb_pbp = pbp_with_ftn.filter(
            (pl.col('rusher_player_id').is_in(rb_players['player_id'].to_list())) &
            (pl.col('play_type') == 'run') &
            (pl.col('is_rpo').is_not_null())  # Ensure FTN data exists
        )
        
        if len(rb_pbp) > 0:
            rb_ftn = rb_pbp.group_by('rusher_player_id').agg([
                pl.len().alias('rush_attempts'),
                pl.col('is_rpo').mean().alias('rpo_rate'),
                (pl.col('n_defense_box') >= 8).mean().alias('heavy_box_rate')
            ]).rename({'rusher_player_id': 'player_id'})
            
            # Join with player names and season scores
            rb_ftn = rb_ftn.join(rb_players, on='player_id', how='left')
            rb_season = season_contributions.filter(pl.col('position') == 'RB').head(15)
            rb_with_ftn = rb_season.join(rb_ftn, on=['player_id', 'player_name'], how='left')
            
            # Filter to RBs with at least 50 rush attempts
            rb_with_ftn = rb_with_ftn.filter(pl.col('rush_attempts') >= 50)
            
            if len(rb_with_ftn) > 0:
                md += "### RB - Scheme Context\n\n"
                md += "*How often RBs face different rushing situations*\n\n"
                md += "- **RPO**: -12% adjustment (easier due to defensive confusion)\n"
                md += "- **8+ Defenders in Box**: +15% adjustment (harder)\n\n"
                
                rb_table = PrettyTable()
                rb_table.field_names = ["Player", "RPO%", "Heavy Box%", "Rush Att", "Net Adjustment"]
                rb_table.align = "l"
                rb_table.float_format = '.1'
                rb_table.set_style(TableStyle.MARKDOWN)
                
                for row in rb_with_ftn.sort('raw_score', descending=True).iter_rows(named=True):
                    player_name = row['player_name']
                    rpo_pct = row['rpo_rate'] * 100 if row['rpo_rate'] is not None else 0
                    heavy_box_pct = row['heavy_box_rate'] * 100 if row['heavy_box_rate'] is not None else 0
                    rush_att = int(row['rush_attempts']) if row['rush_attempts'] is not None else 0
                    
                    # Calculate net adjustment impact
                    # Positive = harder, Negative = easier
                    net_adj = (rpo_pct * -0.12) + (heavy_box_pct * 0.15)
                    adj_display = f"+{net_adj:.1f}%" if net_adj > 0 else f"{net_adj:.1f}%"
                    
                    rb_table.add_row([player_name, f"{rpo_pct:.1f}%", f"{heavy_box_pct:.1f}%", 
                                     rush_att, adj_display])
                
                md += rb_table.get_string() + "\n\n"
    
    # WR/TE Reception Context
    wr_te_players = player_stats.filter(
        pl.col('position').is_in(['WR', 'TE'])
    ).select(['player_id', 'player_name', 'position']).unique()
    
    if len(wr_te_players) > 0:
        wr_te_pbp = pbp_with_ftn.filter(
            (pl.col('receiver_player_id').is_in(wr_te_players['player_id'].to_list())) &
            (pl.col('play_type') == 'pass') &
            (pl.col('is_contested_ball').is_not_null())  # Ensure FTN data exists
        )
        
        if len(wr_te_pbp) > 0:
            wr_te_ftn = wr_te_pbp.group_by('receiver_player_id').agg([
                pl.len().alias('targets'),
                pl.col('is_contested_ball').mean().alias('contested_rate'),
                pl.col('is_drop').sum().alias('drops'),
                pl.col('is_screen_pass').mean().alias('screen_rate')
            ]).rename({'receiver_player_id': 'player_id'})
            
            # Join with player names and season scores
            wr_te_ftn = wr_te_ftn.join(wr_te_players, on='player_id', how='left')
            
            for pos in ['WR', 'TE']:
                pos_season = season_contributions.filter(pl.col('position') == pos).head(15)
                pos_with_ftn = pos_season.join(wr_te_ftn, on=['player_id', 'player_name'], how='left')
                
                # Filter to players with at least 20 targets
                pos_with_ftn = pos_with_ftn.filter(pl.col('targets') >= 20)
                
                if len(pos_with_ftn) > 0:
                    md += f"### {pos} - Reception Context\n\n"
                    md += "*How often receivers face difficult catch situations*\n\n"
                    md += "- **Contested Catch**: +25% adjustment (harder)\n"
                    md += "- **Drop**: -8 pts penalty\n"
                    md += "- **Screen Pass**: -10% adjustment (easier)\n\n"
                    
                    pos_table = PrettyTable()
                    pos_table.field_names = ["Player", "Contested%", "Drops", "Screen%", "Targets", "Net Adjustment"]
                    pos_table.align = "l"
                    pos_table.float_format = '.1'
                    pos_table.set_style(TableStyle.MARKDOWN)
                    
                    for row in pos_with_ftn.sort('raw_score', descending=True).iter_rows(named=True):
                        player_name = row['player_name']
                        contested_pct = row['contested_rate'] * 100 if row['contested_rate'] is not None else 0
                        drops = int(row['drops']) if row['drops'] is not None else 0
                        screen_pct = row['screen_rate'] * 100 if row['screen_rate'] is not None else 0
                        targets = int(row['targets']) if row['targets'] is not None else 0
                        
                        # Calculate net adjustment impact
                        # Positive = harder, Negative = easier
                        net_adj = (contested_pct * 0.25) + (screen_pct * -0.10) + (drops * -0.08)
                        adj_display = f"+{net_adj:.1f}%" if net_adj > 0 else f"{net_adj:.1f}%"
                        
                        pos_table.add_row([player_name, f"{contested_pct:.1f}%", drops, 
                                          f"{screen_pct:.1f}%", targets, adj_display])
                    
                    md += pos_table.get_string() + "\n\n"
    
    md += "*Note: Net Adjustment shows cumulative impact of FTN flags on player scores. Positive = harder situations faced, Negative = easier situations faced.*\n\n"
    
    logger.info(f"Generated FTN context tables for {year}")
    return md


def generate_top_contributors(year: int) -> tuple[str, str]:
    """Generate a table of the top 10 offensive contributors for the season.
    
    Returns:
        tuple of (overview_markdown, deep_dive_markdown)
    """
    logger.info(f"Generating top contributors table for {year}")
    
    # Load team stats
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team data available."
        
    # Load player stats for each position
    position_stats = {}
    for pos in SKILL_POSITIONS:
        df = load_position_weekly_stats(year, pos)
        if df is not None:
            position_stats[pos] = df
    
    if not position_stats:
        return "No player data available."
    
    # Combine all position stats
    player_stats = pl.concat(list(position_stats.values()))
    
    # Apply minimum activity thresholds (1978-now standards)
    # Calculate per-game rates for each player
    player_games = player_stats.group_by(['player_id', 'player_name', 'position']).agg([
        pl.col('week').count().alias('games'),
        pl.col('carries').sum().alias('total_carries'),
        pl.col('receptions').sum().alias('total_receptions')
    ])
    
    # Calculate per-game rates
    player_games = player_games.with_columns([
        (pl.col('total_carries') / pl.col('games')).alias('carries_per_game'),
        (pl.col('total_receptions') / pl.col('games')).alias('receptions_per_game')
    ])
    
    # Apply position-specific minimum thresholds
    # RB: 6.25 rushing attempts per game
    # WR/TE: 1.875 receptions per game
    # Note: QB thresholds (14 attempts/game) handled separately in QB rankings
    qualified_players = player_games.filter(
        ((pl.col('position') == 'RB') & (pl.col('carries_per_game') >= 6.25)) |
        ((pl.col('position').is_in(['WR', 'TE'])) & (pl.col('receptions_per_game') >= 1.875))
    )
    
    # Filter player_stats to only include qualified players
    qualified_ids = qualified_players.select(['player_id', 'player_name']).unique()
    player_stats = player_stats.join(qualified_ids, on=['player_id', 'player_name'], how='inner')
    
    logger.info(f"Filtered to {len(qualified_ids)} qualified players meeting minimum activity thresholds")
    
    # Calculate RAW scores first (without situational adjustments - year=None means multipliers = 1.0)
    raw_contributions = calculate_offensive_shares(team_stats, player_stats, 'overall_contribution', opponent_stats=team_stats, year=None)
    
    # Calculate ADJUSTED scores (with situational adjustments from PBP data)
    contributions = calculate_offensive_shares(team_stats, player_stats, 'overall_contribution', opponent_stats=team_stats, year=year)
    
    # Add position column back by joining with player_stats
    # Ensure player_positions is truly unique per player
    player_positions = player_stats.select(['player_id', 'player_name', 'position']).unique(subset=['player_id', 'player_name'])
    
    # Calculate season aggregates for positional rankings (use RAW scores for fair comparison)
    raw_season_contributions = raw_contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').mean().alias('raw_score')
    ])
    raw_season_contributions = raw_season_contributions.join(player_positions, on=['player_id', 'player_name'], how='left')
    
    # NOTE: Position rankings will be calculated AFTER Phase 5 adjustments
    # to match the rankings shown in position-specific tables
    # See line ~1015 where position_rankings is recalculated
    
    # First, calculate positional rankings (use raw scores for ranking)
    # Rank by (player_id, player_name, team) to handle players with same name on different teams
    position_rankings_old = {}
    for pos in SKILL_POSITIONS:
        pos_stats = raw_season_contributions.filter(
            pl.col('position') == pos
        ).sort('raw_score', descending=True)
        
        position_rankings_old[pos] = {
            (row['player_id'], row['player_name'], row['team']): rank + 1 
            for rank, row in enumerate(pos_stats.iter_rows(named=True))
        }
    
    
    # Add position to contributions for later use
    contributions = contributions.join(player_positions, on=['player_id', 'player_name'], how='left')
    
    # Apply Phase 4 adjustments (catch rate + blocking quality)
    # These are player-level adjustments calculated once per player after aggregation
    contributions = apply_phase4_adjustments(contributions, year)
    
    # Save per-game contributions before Phase 5 aggregation for peak/notable game tracking and consistency metrics
    # Clone the ENTIRE dataframe first, then select columns to ensure we have a deep copy
    per_game_contributions = contributions.clone().select([
        'player_id', 'player_name', 'team', 'position', 'week', 'player_overall_contribution'
    ])
    
    peak_games = (
        per_game_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').max().alias('peak_game_score')
        ])
    )
    
    # Apply Phase 5 adjustments (talent context + sample size dampening)
    # Two-pass system: use baseline scores to calculate teammate quality, then dampen by games played
    contributions = apply_phase5_adjustments(contributions, year)
    
    # Join peak game scores back
    contributions = contributions.join(peak_games, on=['player_id', 'player_name', 'team'], how='left')
    
    # Calculate average difficulty multipliers for all players
    avg_difficulty = calculate_average_difficulty(year, player_stats)
    
    # Initialize markdown strings for both files
    overview_md = f"# Top Contributors - {year}\n\n"
    overview_md += "*High-level overview of top offensive contributors*\n\n"
    overview_md += "## Overall Rankings\n\n"
    
    deep_dive_md = f"# Top Contributors - Deep Dive - {year}\n\n"
    deep_dive_md += "*Detailed analysis with consistency metrics and notable performances*\n\n"
    deep_dive_md += "## Overall Rankings\n\n"
    
    # Create PrettyTable for overview (simplified columns)
    overview_table = PrettyTable()
    overview_table.field_names = ["Rank", "Player", "Team", "Position", "Adjusted Score", 
                                    "Difficulty", "Games", "Avg/Game", "Typical", "Consistency", "Trend"]
    overview_table.align = "l"
    overview_table.float_format = '.2'
    overview_table.set_style(TableStyle.MARKDOWN)
    
    # Create PrettyTable for deep dive (all columns)
    deep_dive_table = PrettyTable()
    deep_dive_table.field_names = ["Rank", "Player", "Team", "Position (Pos. Rank)", "Raw Score", "Adjusted Score", 
                        "Difficulty", "Games", "Avg/Game", "Typical", "Consistency", "Floor", "Ceiling", "Peak", "Trend", "Notable Games"]
    deep_dive_table.align = "l"
    deep_dive_table.float_format = '.2'
    deep_dive_table.set_style(TableStyle.MARKDOWN)
    
    # Get top 10 by adjusted contribution with consistency metrics
    top_contributors = (
        contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').sum().alias('adjusted_score'),  # Total adjusted score
            pl.col('player_overall_contribution').mean().alias('avg_per_game'),  # Average per game
            pl.col('peak_game_score').max().alias('peak_score'),
            pl.col('week').count().alias('games_played')
        ])
        .sort('adjusted_score', descending=True)
        .head(10)
    )
    
    # Calculate consistency metrics from per_game_contributions
    # Need to apply the same Phase 5 scaling (talent + dampening) to make metrics comparable to Avg/Game
    # Strategy: Calculate the scaling factor for each player (Phase 5 adjusted / Phase 4 mean) and apply to all metrics
    
    # Get pre-Phase 5 means for each player
    pre_phase5_means = (
        per_game_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('pre_phase5_mean')
        ])
    )
    
    # Get post-Phase 5 values (dampened) for each player
    post_phase5_values = (
        contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('post_phase5_value')  # All weeks have same value after Phase 5
        ])
    )
    
    # Calculate scaling factor
    scaling_factors = pre_phase5_means.join(post_phase5_values, on=['player_id', 'player_name', 'team'])
    scaling_factors = scaling_factors.with_columns([
        (pl.col('post_phase5_value') / pl.col('pre_phase5_mean')).alias('phase5_scaling_factor')
    ])
    
    # Calculate consistency metrics from per_game_contributions
    consistency_metrics = (
        per_game_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').median().alias('typical_game_raw'),
            pl.col('player_overall_contribution').quantile(0.25).alias('floor_raw'),
            pl.col('player_overall_contribution').quantile(0.75).alias('ceiling_raw')
        ])
    )
    
    # Join scaling factors and apply to all consistency metrics
    consistency_metrics = consistency_metrics.join(scaling_factors, on=['player_id', 'player_name', 'team'], how='left')
    consistency_metrics = consistency_metrics.with_columns([
        (pl.col('typical_game_raw') * pl.col('phase5_scaling_factor')).alias('typical_game'),
        (pl.col('floor_raw') * pl.col('phase5_scaling_factor')).alias('floor'),
        (pl.col('ceiling_raw') * pl.col('phase5_scaling_factor')).alias('ceiling')
    ])
    
    # Calculate consistency profile (below/at/above average games)
    # Join typical_game back to per_game data to count performance levels
    per_game_with_typical = per_game_contributions.join(
        consistency_metrics.select(['player_id', 'player_name', 'team', 'typical_game']),
        on=['player_id', 'player_name', 'team'],
        how='left'
    )
    
    consistency_profile = (
        per_game_with_typical.group_by(['player_id', 'player_name', 'team'])
        .agg([
            # Below: < 95% of typical
            (pl.col('player_overall_contribution') < (pl.col('typical_game') * 0.95)).sum().alias('below_avg'),
            # At: 95-105% of typical
            ((pl.col('player_overall_contribution') >= (pl.col('typical_game') * 0.95)) & 
             (pl.col('player_overall_contribution') <= (pl.col('typical_game') * 1.05))).sum().alias('at_avg'),
            # Above: > 105% of typical
            (pl.col('player_overall_contribution') > (pl.col('typical_game') * 1.05)).sum().alias('above_avg')
        ])
    )
    
    # Join consistency profile to consistency metrics
    consistency_metrics = consistency_metrics.join(consistency_profile, on=['player_id', 'player_name', 'team'], how='left')
    
    # Select metrics for joining (removed std_dev, added consistency profile)
    consistency_metrics = consistency_metrics.select([
        'player_id', 'player_name', 'team', 'typical_game', 'floor', 'ceiling', 
        'below_avg', 'at_avg', 'above_avg'
    ])
    
    top_contributors = top_contributors.join(consistency_metrics, on=['player_id', 'player_name', 'team'], how='left')
    
    # Recalculate positional rankings based on ADJUSTED scores (after Phase 5)
    # This ensures the position ranks in "Overall Rankings" match the position-specific tables
    adjusted_season_contributions = contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').sum().alias('adjusted_score'),  # Total adjusted score
        pl.col('position').first().alias('position')  # Carry position through
    ])
    
    position_rankings = {}
    for pos in SKILL_POSITIONS:
        pos_stats = adjusted_season_contributions.filter(
            pl.col('position') == pos
        ).sort('adjusted_score', descending=True)
        
        position_rankings[pos] = {
            (row['player_id'], row['player_name'], row['team']): rank + 1 
            for rank, row in enumerate(pos_stats.iter_rows(named=True))
        }
    
    # Add raw scores from raw_contributions
    raw_scores = (
        raw_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').sum().alias('raw_score')  # Total raw score
        ])
    )
    top_contributors = top_contributors.join(raw_scores, on=['player_id', 'player_name', 'team'], how='left')
    
    # Join difficulty metrics
    if avg_difficulty is not None:
        top_contributors = top_contributors.join(
            avg_difficulty.select(['player_id', 'player_name', 'avg_difficulty_multiplier']),
            on=['player_id', 'player_name'],
            how='left'
        )
    
    # Get notable games (weeks where player had >150% of their average contribution)
    # Need to get opponent info from player_stats since it's not in contributions
    notable_games = {}
    for row in top_contributors.iter_rows(named=True):
        player_id = row['player_id']
        player_name = row['player_name']
        team_name = row['team']
        typical_game = row['typical_game']  # Use median for notable games threshold
        
        # Get player weeks with opponent information from player_stats
        player_weeks = player_stats.filter(
            (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name) & (pl.col('team') == team_name)
        ).select(['week', 'opponent_team']).unique()
        
        # Get per-game contribution scores (before Phase 5 aggregation)
        player_contrib = per_game_contributions.filter(
            (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name) & (pl.col('team') == team_name)
        ).select(['week', 'player_overall_contribution'])
        
        # Join to get both opponent and contribution
        player_full = player_contrib.join(player_weeks, on='week', how='left')
        
        high_games = player_full.filter(
            pl.col('player_overall_contribution') > typical_game * 1.5
        ).sort('player_overall_contribution', descending=True).head(3)
        
        if len(high_games) > 0:
            # Format as "Wk # (vs OPP)"
            game_strings = []
            for game_row in high_games.iter_rows(named=True):
                week_num = game_row['week']
                opponent = game_row.get('opponent_team', '?')
                game_strings.append(f"Wk {week_num} (vs {opponent})")
            notable_games[player_name] = ", ".join(game_strings)
        else:
            notable_games[player_name] = ""
    
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Position (Pos. Rank)", "Raw Score", "Adjusted Score", 
                        "Difficulty", "Games", "Avg/Game", "Typical", "Std Dev", "Floor", "Ceiling", "Peak", "Trend", "Notable Games"]
    table.align = "l"  # Left align text
    table.float_format = '.2'  # Two decimal places for floats
    
    for rank, row in enumerate(top_contributors.iter_rows(named=True), 1):
        player_id = row['player_id']
        player_name = row['player_name']
        team = row['team']
        position = player_stats.filter((pl.col('player_id') == player_id) & (pl.col('player_name') == player_name))['position'].head(1)[0]
        pos_rank = position_rankings[position].get((player_id, player_name, team), "N/A")
        position_display = f"{position} (#{pos_rank})"
        position_only = position
        raw_score = row['raw_score']
        adj_score_total = row['adjusted_score']  # Now the total adjusted score
        games_played = row['games_played']
        avg_per_game = row['avg_per_game']  # Now separate from adjusted_score
        peak = f"{row['peak_score']:.2f}"
        
        # Calculate trend (compare first half to second half median - more robust to outliers)
        # Use per_game_contributions (before Phase 5) to get actual weekly performance
        player_games = per_game_contributions.filter(
            (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name)
        ).sort('week')
        if len(player_games) >= 4:
            mid_point = len(player_games) // 2
            first_half_median = player_games.head(mid_point)['player_overall_contribution'].median()
            second_half_median = player_games.tail(len(player_games) - mid_point)['player_overall_contribution'].median()
            if second_half_median > first_half_median * 1.15:  # Stricter threshold for median
                trend = "Increasing"
            elif second_half_median < first_half_median * 0.85:  # Stricter threshold for median
                trend = "Decreasing"
            else:
                trend = "Stable"
        else:
            trend = "Stable"
        
        games = notable_games.get(player_name, "")
        
        # Get consistency metrics
        typical = row.get('typical_game', avg_per_game)
        floor = row.get('floor', 0)
        ceiling = row.get('ceiling', 0)
        below_avg = row.get('below_avg', 0)
        at_avg = row.get('at_avg', 0)
        above_avg = row.get('above_avg', 0)
        consistency = f"{below_avg}/{at_avg}/{above_avg}"
        
        # Get difficulty multiplier
        difficulty_mult = row.get('avg_difficulty_multiplier', 1.0)
        difficulty_str = f"{difficulty_mult:.3f}" if difficulty_mult else "1.000"
        
        # Add to overview table (simplified)
        overview_table.add_row([rank, player_name, team, position_only, f"{adj_score_total:.2f}", 
                                difficulty_str, games_played, f"{avg_per_game:.2f}", f"{typical:.2f}", 
                                consistency, trend])
        
        # Add to deep dive table (full details)
        deep_dive_table.add_row([rank, player_name, team, position_display, f"{raw_score:.2f}", f"{adj_score_total:.2f}", 
                                 difficulty_str, games_played, f"{avg_per_game:.2f}", f"{typical:.2f}", consistency, 
                                 f"{floor:.1f}", f"{ceiling:.1f}", peak, trend, games])
    
    overview_md += overview_table.get_string() + "\n\n"
    deep_dive_md += deep_dive_table.get_string() + "\n\n"
    
    # Add positional rankings for RB, WR, TE
    for pos in ['RB', 'WR', 'TE']:
        overview_md += f"## {pos} Rankings\n\n"
        deep_dive_md += f"## {pos} Rankings\n\n"
        
        # Get top 10 players at this position
        pos_contributors = (
            contributions.filter(pl.col('position') == pos)
            .group_by(['player_id', 'player_name', 'team'])
            .agg([
                pl.col('player_overall_contribution').sum().alias('adjusted_score'),  # Total adjusted score
                pl.col('player_overall_contribution').mean().alias('avg_per_game'),  # Average per game
                pl.col('peak_game_score').max().alias('peak_score'),
                pl.col('week').count().alias('games_played')
            ])
            .sort('adjusted_score', descending=True)
            .head(10)
        )
        
        # Add consistency metrics for this position
        # Apply Phase 5 scaling to make metrics comparable to Avg/Game
        pos_per_game = per_game_contributions.filter(pl.col('position') == pos)
        
        # Get pre-Phase 5 means for this position
        pos_pre_phase5_means = (
            pos_per_game.group_by(['player_id', 'player_name', 'team'])
            .agg([
                pl.col('player_overall_contribution').mean().alias('pre_phase5_mean')
            ])
        )
        
        # Get post-Phase 5 values (already calculated in scaling_factors from above)
        # Join with the overall scaling_factors we calculated earlier
        pos_consistency = (
            pos_per_game.group_by(['player_id', 'player_name', 'team'])
            .agg([
                pl.col('player_overall_contribution').median().alias('typical_game_raw'),
                pl.col('player_overall_contribution').quantile(0.25).alias('floor_raw'),
                pl.col('player_overall_contribution').quantile(0.75).alias('ceiling_raw')
            ])
        )
        
        # Join scaling factors and apply
        pos_consistency = pos_consistency.join(scaling_factors.select(['player_id', 'player_name', 'team', 'phase5_scaling_factor']), 
                                               on=['player_id', 'player_name', 'team'], how='left')
        pos_consistency = pos_consistency.with_columns([
            (pl.col('typical_game_raw') * pl.col('phase5_scaling_factor')).alias('typical_game'),
            (pl.col('floor_raw') * pl.col('phase5_scaling_factor')).alias('floor'),
            (pl.col('ceiling_raw') * pl.col('phase5_scaling_factor')).alias('ceiling')
        ])
        
        # Calculate consistency profile for position
        pos_per_game_with_typical = pos_per_game.join(
            pos_consistency.select(['player_id', 'player_name', 'team', 'typical_game']),
            on=['player_id', 'player_name', 'team'],
            how='left'
        )
        
        pos_consistency_profile = (
            pos_per_game_with_typical.group_by(['player_id', 'player_name', 'team'])
            .agg([
                (pl.col('player_overall_contribution') < (pl.col('typical_game') * 0.95)).sum().alias('below_avg'),
                ((pl.col('player_overall_contribution') >= (pl.col('typical_game') * 0.95)) & 
                 (pl.col('player_overall_contribution') <= (pl.col('typical_game') * 1.05))).sum().alias('at_avg'),
                (pl.col('player_overall_contribution') > (pl.col('typical_game') * 1.05)).sum().alias('above_avg')
            ])
        )
        
        pos_consistency = pos_consistency.join(pos_consistency_profile, on=['player_id', 'player_name', 'team'], how='left')
        
        pos_consistency = pos_consistency.select([
            'player_id', 'player_name', 'team', 'typical_game', 'floor', 'ceiling',
            'below_avg', 'at_avg', 'above_avg'
        ])
        
        pos_contributors = pos_contributors.join(pos_consistency, on=['player_id', 'player_name', 'team'], how='left')
        
        # Add raw scores
        pos_contributors = pos_contributors.join(raw_scores, on=['player_id', 'player_name', 'team'], how='left')
        
        # Add difficulty metrics
        if avg_difficulty is not None:
            pos_contributors = pos_contributors.join(
                avg_difficulty.select(['player_id', 'player_name', 'avg_difficulty_multiplier']),
                on=['player_id', 'player_name'],
                how='left'
            )
        
        # Create overview table for this position (simplified)
        pos_overview_table = PrettyTable()
        pos_overview_table.field_names = ["Rank", "Player", "Team", "Adjusted Score", 
                                          "Difficulty", "Games", "Avg/Game", "Typical", "Consistency", "Trend"]
        pos_overview_table.align = "l"
        pos_overview_table.float_format = '.2'
        pos_overview_table.set_style(TableStyle.MARKDOWN)
        
        # Create deep dive table for this position (all columns)
        pos_deep_dive_table = PrettyTable()
        pos_deep_dive_table.field_names = ["Rank", "Player", "Team", "Raw Score", "Adjusted Score", 
                                "Difficulty", "Games", "Avg/Game", "Typical", "Consistency", "Floor", "Ceiling", "Peak", "Trend", "Notable Games (>150% Typical)"]
        pos_deep_dive_table.align = "l"
        pos_deep_dive_table.float_format = '.2'
        pos_deep_dive_table.set_style(TableStyle.MARKDOWN)
        
        for rank, row in enumerate(pos_contributors.iter_rows(named=True), 1):
            player_id = row['player_id']
            player_name = row['player_name']
            team = row['team']
            raw_score = row['raw_score']
            adj_score_total = row['adjusted_score']  # Now the total adjusted score
            games_played = row['games_played']
            avg_per_game = row['avg_per_game']  # Now separate from adjusted_score
            peak = f"{row['peak_score']:.2f}"
            
            # Calculate trend using median (more robust to outliers)
            player_games = per_game_contributions.filter(
                (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name) & (pl.col('team') == team)
            ).sort('week')
            if len(player_games) >= 4:
                mid_point = len(player_games) // 2
                first_half_median = player_games.head(mid_point)['player_overall_contribution'].median()
                second_half_median = player_games.tail(len(player_games) - mid_point)['player_overall_contribution'].median()
                if second_half_median > first_half_median * 1.15:
                    trend = "Increasing"
                elif second_half_median < first_half_median * 0.85:
                    trend = "Decreasing"
                else:
                    trend = "Stable"
            else:
                trend = "Stable"
            
            # Get notable games for this player (reuse from notable_games dict or recalculate)
            # Use (player_id, player_name) key since that's unique
            games = notable_games.get(player_name, "")
            
            # Get consistency metrics
            typical = row.get('typical_game', avg_per_game)
            floor = row.get('floor', 0)
            ceiling = row.get('ceiling', 0)
            below_avg = row.get('below_avg', 0)
            at_avg = row.get('at_avg', 0)
            above_avg = row.get('above_avg', 0)
            consistency = f"{below_avg}/{at_avg}/{above_avg}"
            
            # Get difficulty multiplier
            difficulty_mult = row.get('avg_difficulty_multiplier', 1.0)
            difficulty_str = f"{difficulty_mult:.3f}" if difficulty_mult else "1.000"
            
            # Add to overview table (simplified)
            pos_overview_table.add_row([rank, player_name, team, f"{adj_score_total:.2f}", 
                                        difficulty_str, games_played, f"{avg_per_game:.2f}", 
                                        f"{typical:.2f}", consistency, trend])
            
            # Add to deep dive table (full details)
            pos_deep_dive_table.add_row([rank, player_name, team, f"{raw_score:.2f}", f"{adj_score_total:.2f}", 
                                         difficulty_str, games_played, f"{avg_per_game:.2f}", f"{typical:.2f}", consistency,
                                         f"{floor:.1f}", f"{ceiling:.1f}", peak, trend, games])
        
        overview_md += pos_overview_table.get_string() + "\n\n"
        deep_dive_md += pos_deep_dive_table.get_string() + "\n\n"
    
    # Add difficulty context section (only to deep dive)
    difficulty_context = calculate_difficulty_context(year, player_stats)
    if difficulty_context is not None:
        deep_dive_md += "## Difficulty Context\n\n"
        deep_dive_md += "*How challenging were the situations these players faced?*\n\n"
        
        # Create tables for each position group
        for pos in ['RB', 'WR', 'TE']:
            pos_difficulty = difficulty_context.filter(pl.col('position') == pos)
            if len(pos_difficulty) == 0:
                continue
            
            # Join with season contributions to get top players
            pos_season = raw_season_contributions.filter(pl.col('position') == pos).head(15)
            pos_with_difficulty = pos_season.join(
                pos_difficulty, 
                on=['player_id', 'player_name'], 
                how='left'
            ).sort('raw_score', descending=True)
            
            if pos == 'RB':
                # Filter to players with at least 50 rush attempts
                pos_with_difficulty = pos_with_difficulty.filter(
                    pl.col('rush_attempts') >= 50
                )
                
                if len(pos_with_difficulty) == 0:
                    continue
                
                deep_dive_md += f"### {pos} - Box Count Context\n\n"
                deep_dive_md += "*Higher box counts = tougher runs (8+ defenders = stacked box)*\n\n"
                
                diff_table = PrettyTable()
                diff_table.field_names = ["Player", "Avg Box", "8+ Box %", "Avg Multiplier", "Rush Att"]
                diff_table.align = "l"
                diff_table.float_format = '.2'
                diff_table.set_style(TableStyle.MARKDOWN)
                
                for row in pos_with_difficulty.iter_rows(named=True):
                    if row.get('avg_defenders_in_box') is not None:
                        player_name = row['player_name']
                        avg_box = row['avg_defenders_in_box']
                        pct_8plus = row['pct_vs_8plus_box'] * 100
                        avg_mult = row['avg_box_multiplier']
                        rush_att = int(row['rush_attempts'])
                        diff_table.add_row([player_name, f"{avg_box:.1f}", f"{pct_8plus:.1f}%", 
                                          f"{avg_mult:.3f}", rush_att])
                
                deep_dive_md += diff_table.get_string() + "\n\n"
            
            else:  # WR or TE
                # Filter to players with at least 20 targets
                pos_with_difficulty = pos_with_difficulty.filter(
                    pl.col('targets') >= 20
                )
                
                if len(pos_with_difficulty) == 0:
                    continue
                
                deep_dive_md += f"### {pos} - Coverage Context\n\n"
                deep_dive_md += "*Higher man coverage % = tougher matchups (2-Man, 1-Man, 0-Man, Cover-1)*\n\n"
                
                diff_table = PrettyTable()
                diff_table.field_names = ["Player", "Man Cov %", "Avg Multiplier", "Targets"]
                diff_table.align = "l"
                diff_table.float_format = '.2'
                diff_table.set_style(TableStyle.MARKDOWN)
                
                for row in pos_with_difficulty.iter_rows(named=True):
                    if row.get('pct_vs_man_coverage') is not None:
                        player_name = row['player_name']
                        pct_man = row['pct_vs_man_coverage'] * 100
                        avg_mult = row['avg_coverage_multiplier']
                        targets = int(row['targets'])
                        diff_table.add_row([player_name, f"{pct_man:.1f}%", 
                                          f"{avg_mult:.3f}", targets])
                
                deep_dive_md += diff_table.get_string() + "\n\n"
        
        deep_dive_md += "*Note: Multipliers > 1.0 indicate tougher situations (bonus), < 1.0 indicate easier situations (penalty)*\n\n"
    
    # Add FTN context section (only for years 2022+)
    from modules.ftn_cache_builder import FTN_START_YEAR
    if year >= FTN_START_YEAR:
        ftn_context = generate_ftn_context(year, player_stats, raw_season_contributions)
        if ftn_context:
            deep_dive_md += ftn_context
    
    return overview_md, deep_dive_md

def process_year(year: int) -> tuple[bool, str, str, str, str, str, str, str, str]:
    """Process a single year's data with error recovery.
    
    Returns:
        tuple of (success, weekly_markdown, summary_markdown, top_contributors_overview_markdown, 
                 top_contributors_deep_dive_markdown, qb_rankings_markdown, rb_rankings_markdown,
                 wr_rankings_markdown, te_rankings_markdown)
    """
    try:
        logger.info(f"Processing year {year}")
        weekly_markdown = generate_weekly_tables(year)
        summary_markdown = generate_season_summary(year)
        top_contributors_overview, top_contributors_deep_dive = generate_top_contributors(year)
        qb_rankings = generate_qb_rankings(year)
        rb_rankings = generate_rb_rankings(year)
        wr_rankings = generate_wr_rankings(year)
        te_rankings = generate_te_rankings(year)
        return True, weekly_markdown, summary_markdown, top_contributors_overview, top_contributors_deep_dive, qb_rankings, rb_rankings, wr_rankings, te_rankings
    except Exception as e:
        logger.error(f"Failed to process year {year}: {str(e)}")
        error_msg = f"Error processing {year}: {str(e)}"
        return False, error_msg, error_msg, error_msg, error_msg, error_msg, error_msg, error_msg, error_msg

def check_and_rebuild_caches(years: list[int], parallel: bool = True) -> None:
    """Check all years for missing caches and rebuild all cache types as needed.

    This runs before main processing to ensure all caches exist and have required data.
    Rebuilds PBP, positional player stats, team stats, FTN, injury, penalty, and weather caches.

    Args:
        years: List of years to check
        parallel: Whether to rebuild in parallel (default True)
    """
    from modules.positional_cache_builder import build_positional_cache_for_year
    from modules.team_cache_builder import build_team_cache_for_year
    from modules.ftn_cache_builder import build_ftn_cache_for_year, FTN_START_YEAR
    from modules.injury_cache_builder import build_injury_cache, INJURY_DATA_START_YEAR
    from modules.penalty_cache_builder import build_penalty_cache_for_year
    from modules.weather_cache_builder import build_weather_performance_cache

    logger.info("Checking cache completeness for all years...")

    required_cols = ['defenders_in_box', 'defense_coverage_type',
                     'defenders_in_box_multiplier', 'coverage_multiplier']

    years_needing_pbp_rebuild = []
    years_needing_positional_rebuild = []
    years_needing_ftn_rebuild = []
    years_needing_team_rebuild = []
    years_needing_injury_rebuild = []
    years_needing_penalty_rebuild = []
    years_needing_weather_rebuild = []
    
    # Check each year's caches
    for year in years:
        # Check PBP cache
        pbp_path = Path("cache/pbp") / f"pbp_{year}.parquet"
        if not pbp_path.exists():
            logger.info(f"PBP cache missing for {year}, will rebuild")
            years_needing_pbp_rebuild.append(year)
        else:
            try:
                pbp_data = pl.read_parquet(pbp_path)
                if not all(col in pbp_data.columns for col in required_cols):
                    logger.info(f"PBP cache for {year} missing difficulty columns, will rebuild")
                    years_needing_pbp_rebuild.append(year)
            except Exception as e:
                logger.warning(f"Error reading cache for {year}: {e}, will rebuild")
                years_needing_pbp_rebuild.append(year)
        
        # Check positional player stats cache
        positional_cache_dir = Path("cache/positional_player_stats")
        if not positional_cache_dir.exists():
            years_needing_positional_rebuild.append(year)
        else:
            # Check if we have at least some position data for this year
            has_positional_data = False
            for pos_dir in positional_cache_dir.iterdir():
                if pos_dir.is_dir():
                    pos_file = pos_dir / f"{pos_dir.name}-{year}.csv"
                    if pos_file.exists():
                        has_positional_data = True
                        break
            if not has_positional_data:
                logger.info(f"Positional player stats cache missing for {year}, will rebuild")
                years_needing_positional_rebuild.append(year)
        
        # Check team stats cache
        team_cache_dir = Path("cache/team_stats")
        if not team_cache_dir.exists():
            years_needing_team_rebuild.append(year)
        else:
            # Check if we have at least some team data for this year
            has_team_data = False
            for team_dir in team_cache_dir.iterdir():
                if team_dir.is_dir():
                    team_file = team_dir / f"{team_dir.name}-{year}.csv"
                    if team_file.exists():
                        has_team_data = True
                        break
            if not has_team_data:
                logger.info(f"Team stats cache missing for {year}, will rebuild")
                years_needing_team_rebuild.append(year)
        
        # Check FTN cache (only for 2022+)
        if year >= FTN_START_YEAR:
            ftn_path = Path("cache/ftn") / f"ftn_{year}.parquet"
            if not ftn_path.exists():
                logger.info(f"FTN cache missing for {year}, will rebuild")
                years_needing_ftn_rebuild.append(year)
        
        # Check injury cache (only for 2009+)
        if year >= INJURY_DATA_START_YEAR:
            injury_path = Path("cache/injuries") / f"injuries-{year}.csv"
            if not injury_path.exists():
                logger.info(f"Injury cache missing for {year}, will rebuild")
                years_needing_injury_rebuild.append(year)
        
        # Check penalty cache
        penalty_path = Path("cache/penalties") / f"penalties-{year}.csv"
        if not penalty_path.exists():
            logger.info(f"Penalty cache missing for {year}, will rebuild")
            years_needing_penalty_rebuild.append(year)

        # Check weather cache (all 4 positions required)
        weather_cache_dir = Path("cache/weather")
        positions = ['QB', 'RB', 'WR', 'TE']
        missing_weather = False
        for pos in positions:
            weather_path = weather_cache_dir / f"weather_{pos.lower()}_{year}.parquet"
            if not weather_path.exists():
                missing_weather = True
                break
        if missing_weather:
            logger.info(f"Weather cache missing for {year}, will rebuild")
            years_needing_weather_rebuild.append(year)

    total_rebuilds = len(set(years_needing_pbp_rebuild + years_needing_positional_rebuild + years_needing_team_rebuild + years_needing_ftn_rebuild + years_needing_injury_rebuild + years_needing_penalty_rebuild + years_needing_weather_rebuild))
    
    if total_rebuilds == 0:
        logger.info("All caches up to date!")
        return
    
    logger.info(f"Rebuilding caches for {total_rebuilds} years")
    if years_needing_pbp_rebuild:
        logger.info(f"  PBP: {years_needing_pbp_rebuild}")
    if years_needing_positional_rebuild:
        logger.info(f"  Positional: {years_needing_positional_rebuild}")
    if years_needing_team_rebuild:
        logger.info(f"  Team: {years_needing_team_rebuild}")
    if years_needing_ftn_rebuild:
        logger.info(f"  FTN: {years_needing_ftn_rebuild}")
    if years_needing_injury_rebuild:
        logger.info(f"  Injury: {years_needing_injury_rebuild}")
    if years_needing_penalty_rebuild:
        logger.info(f"  Penalty: {years_needing_penalty_rebuild}")
    if years_needing_weather_rebuild:
        logger.info(f"  Weather: {years_needing_weather_rebuild}")

    # Rebuild all cache types
    years_to_rebuild = sorted(set(years_needing_pbp_rebuild + years_needing_positional_rebuild + years_needing_team_rebuild + years_needing_ftn_rebuild + years_needing_injury_rebuild + years_needing_penalty_rebuild + years_needing_weather_rebuild))
    
    # Handle injury and penalty caches separately (they rebuild in bulk)
    if years_needing_injury_rebuild:
        try:
            logger.info(f"Rebuilding injury cache for years {min(years_needing_injury_rebuild)}-{max(years_needing_injury_rebuild)}...")
            build_injury_cache(min(years_needing_injury_rebuild), max(years_needing_injury_rebuild))
            logger.info("Injury cache rebuilt successfully")
        except Exception as e:
            logger.warning(f"Injury cache rebuild incomplete: {e}")
    
    if years_needing_penalty_rebuild:
        try:
            logger.info(f"Rebuilding penalty cache for years {min(years_needing_penalty_rebuild)}-{max(years_needing_penalty_rebuild)}...")
            for year in range(min(years_needing_penalty_rebuild), max(years_needing_penalty_rebuild) + 1):
                build_penalty_cache_for_year(year)
            logger.info("Penalty cache rebuilt successfully")
        except Exception as e:
            logger.warning(f"Penalty cache rebuild incomplete: {e}")

    if years_needing_weather_rebuild:
        try:
            logger.info(f"Rebuilding weather cache for years {min(years_needing_weather_rebuild)}-{max(years_needing_weather_rebuild)}...")
            for year in years_needing_weather_rebuild:
                for position in ['QB', 'RB', 'WR', 'TE']:
                    build_weather_performance_cache(year, position)
            logger.info("Weather cache rebuilt successfully")
        except Exception as e:
            logger.warning(f"Weather cache rebuild incomplete: {e}")

    if parallel and len(years_to_rebuild) > 1:
        from concurrent.futures import ThreadPoolExecutor
        logger.info(f"Rebuilding {len(years_to_rebuild)} caches in parallel...")
        with ThreadPoolExecutor(max_workers=min(len(years_to_rebuild), 8)) as executor:
            futures = {}
            for year in years_to_rebuild:
                if year in years_needing_pbp_rebuild:
                    futures[executor.submit(build_cache, year, True)] = (year, "PBP")
                if year in years_needing_positional_rebuild:
                    futures[executor.submit(build_positional_cache_for_year, year)] = (year, "Positional")
                if year in years_needing_team_rebuild:
                    futures[executor.submit(build_team_cache_for_year, year)] = (year, "Team")
                if year in years_needing_ftn_rebuild:
                    futures[executor.submit(build_ftn_cache_for_year, year)] = (year, "FTN")
            
            for future in concurrent.futures.as_completed(futures):
                year, cache_type = futures[future]
                try:
                    result = future.result()
                    if cache_type == "PBP" and result:
                        logger.info(f"{cache_type} cache rebuilt for {year}")
                    elif cache_type in ["Positional", "Team", "FTN"]:
                        logger.info(f"{cache_type} cache rebuilt for {year}")
                    else:
                        logger.warning(f"{cache_type} cache rebuild failed for {year}")
                except Exception as e:
                    logger.error(f"{cache_type} cache rebuild error for {year}: {e}")
    else:
        # Sequential rebuild
        for year in years_to_rebuild:
            if year in years_needing_pbp_rebuild:
                try:
                    logger.info(f"Rebuilding PBP cache for {year}...")
                    success = build_cache(year, force=True)
                    if success:
                        logger.info(f"PBP cache rebuilt for {year}")
                    else:
                        logger.warning(f"PBP cache rebuild failed for {year}")
                except Exception as e:
                    logger.error(f"PBP cache rebuild error for {year}: {e}")
            
            if year in years_needing_positional_rebuild:
                try:
                    logger.info(f"Rebuilding positional cache for {year}...")
                    build_positional_cache_for_year(year)
                    logger.info(f"Positional cache rebuilt for {year}")
                except Exception as e:
                    logger.error(f"Positional cache rebuild error for {year}: {e}")
            
            if year in years_needing_team_rebuild:
                try:
                    logger.info(f"Rebuilding team cache for {year}...")
                    build_team_cache_for_year(year)
                    logger.info(f"Team cache rebuilt for {year}")
                except Exception as e:
                    logger.error(f"Team cache rebuild error for {year}: {e}")
            
            if year in years_needing_ftn_rebuild:
                try:
                    logger.info(f"Rebuilding FTN cache for {year}...")
                    build_ftn_cache_for_year(year)
                    logger.info(f"FTN cache rebuilt for {year}")
                except Exception as e:
                    logger.error(f"FTN cache rebuild error for {year}: {e}")
    
    logger.info("Cache rebuild complete!")

def main(start_year: int = None, end_year: int = None, parallel: bool = True):
    """Generate offensive share analysis for a range of years or all available years.
    
    Args:
        start_year: Start year for analysis (defaults to START_YEAR constant)
        end_year: End year for analysis (defaults to END_YEAR constant)
        parallel: Whether to use parallel processing (default True)
    """
    if start_year is None:
        start_year = START_YEAR
    if end_year is None:
        end_year = END_YEAR
        
    logger.info(f"Starting offensive share analysis for years {start_year}-{end_year}")
    
    # Create base output directory
    output_base = Path("output")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process each year
    years = list(range(start_year, end_year + 1))
    
    # Check and rebuild caches upfront if needed
    check_and_rebuild_caches(years, parallel=parallel)
    
    results = {}
    
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(years), 8)) as executor:
            future_to_year = {executor.submit(process_year, year): year for year in years}
            for future in concurrent.futures.as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    success, weekly, summary, top_overview, top_deep_dive, qb_rankings, rb_rankings, wr_rankings, te_rankings = future.result()
                    results[year] = (success, weekly, summary, top_overview, top_deep_dive, qb_rankings, rb_rankings, wr_rankings, te_rankings)
                except Exception as e:
                    logger.error(f"Year {year} failed with error: {str(e)}")
                    results[year] = (False, str(e), str(e), str(e), str(e), str(e), str(e), str(e), str(e))
    else:
        for year in years:
            success, weekly, summary, top_overview, top_deep_dive, qb_rankings, rb_rankings, wr_rankings, te_rankings = process_year(year)
            results[year] = (success, weekly, summary, top_overview, top_deep_dive, qb_rankings, rb_rankings, wr_rankings, te_rankings)
    
    # Save results and build index
    successful_years = []
    failed_years = []
    
    for year, (success, weekly, summary, top_overview, top_deep_dive, qb_rankings, rb_rankings, wr_rankings, te_rankings) in results.items():
        output_dir = output_base / str(year)
        output_dir.mkdir(exist_ok=True)
        
        if success:
            successful_years.append(year)
            weekly_file = output_dir / "weekly_analysis.md"
            summary_file = output_dir / "season_summary.md"
            top_contributors_file = output_dir / "top_contributors.md"
            top_contributors_deep_dive_file = output_dir / "top_contributors_deep_dive.md"
            qb_rankings_file = output_dir / "qb_rankings.md"
            rb_rankings_file = output_dir / "rb_rankings.md"
            wr_rankings_file = output_dir / "wr_rankings.md"
            te_rankings_file = output_dir / "te_rankings.md"
            weekly_file.write_text(weekly, encoding='utf-8')
            summary_file.write_text(summary, encoding='utf-8')
            top_contributors_file.write_text(top_overview, encoding='utf-8')
            top_contributors_deep_dive_file.write_text(top_deep_dive, encoding='utf-8')
            qb_rankings_file.write_text(qb_rankings, encoding='utf-8')
            rb_rankings_file.write_text(rb_rankings, encoding='utf-8')
            wr_rankings_file.write_text(wr_rankings, encoding='utf-8')
            te_rankings_file.write_text(te_rankings, encoding='utf-8')
            logger.info(f"Saved analysis for {year}")
        else:
            failed_years.append(year)
            error_file = output_dir / "error.md"
            error_file.write_text(f"# Error Processing {year}\n\n{weekly}\n")
            logger.error(f"Failed to process {year}")
    
    # Generate index with status indicators
    index_markdown = "# NFL Offensive Share Analysis\n\n"
    
    if successful_years:
        index_markdown += "## Successfully Processed Years\n\n"
        for year in sorted(successful_years):
            index_markdown += f"- [{year}](./{year}/)\n"
            index_markdown += f"  - [Weekly Analysis](./{year}/weekly_analysis.md)\n"
            index_markdown += f"  - [Season Summary](./{year}/season_summary.md)\n"
            index_markdown += f"  - [Top Contributors - Overview](./{year}/top_contributors.md)\n"
            index_markdown += f"  - [Top Contributors - Deep Dive](./{year}/top_contributors_deep_dive.md)\n"
            index_markdown += f"  - [QB Rankings](./{year}/qb_rankings.md)\n"
            index_markdown += f"  - [RB Rankings](./{year}/rb_rankings.md)\n"
            index_markdown += f"  - [WR Rankings](./{year}/wr_rankings.md)\n"
            index_markdown += f"  - [TE Rankings](./{year}/te_rankings.md)\n\n"
    
    if failed_years:
        index_markdown += "## Failed Years\n\n"
        for year in sorted(failed_years):
            index_markdown += f"- [{year}](./{year}/error.md)\n\n"
    
    index_file = output_base / "index.md"
    index_file.write_text(index_markdown, encoding='utf-8')
    logger.info(f"Saved multi-year index to {index_file}")
    
    if failed_years:
        logger.warning(f"Failed to process years: {failed_years}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NFL offensive share analysis")
    parser.add_argument("year", type=int, nargs='?', help="Single year to process (optional)")
    parser.add_argument("--start-year", type=int, help="Start year (default: START_YEAR from constants)")
    parser.add_argument("--end-year", type=int, help="End year (default: END_YEAR from constants)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    args = parser.parse_args()
    
    # If single year provided, use it for both start and end
    if args.year:
        start_year = args.year
        end_year = args.year
    else:
        start_year = args.start_year
        end_year = args.end_year
    
    main(start_year, end_year, not args.no_parallel)
