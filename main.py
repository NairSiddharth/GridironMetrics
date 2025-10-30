"""
main.py

Analyzes NFL offensive skill player shares on a weekly and seasonal basis.
Generates Markdown formatted tables showing offensive contribution percentages.
"""

from pathlib import Path
import polars as pl
from prettytable import PrettyTable, TableStyle
from modules.logger import get_logger
from modules.constants import START_YEAR, END_YEAR, CACHE_DIR
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

def load_team_weekly_stats(year: int) -> pl.DataFrame:
    """Load team weekly stats for a given year."""
    try:
        # Get all team data from the cache
        team_data = []
        team_cache_dir = Path(CACHE_DIR) / "team_stats"
        for team_dir in team_cache_dir.iterdir():
            if team_dir.is_dir():
                file_path = team_dir / f"{team_dir.name}-{year}.csv"
                if file_path.exists():
                    try:
                        df = pl.read_csv(file_path, infer_schema_length=10000)
                        team_data.append(df)
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
                          'receptions', 'targets', 'carries', 'sacks', 'interceptions']
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

def adjust_for_opponent(df: pl.DataFrame, opponent_stats: pl.DataFrame) -> pl.DataFrame:
    """Apply opponent strength adjustments based on defensive rankings."""
    # Join with opponent defensive rankings
    df_with_opponent = df.join(
        opponent_stats.select([
            "team",
            "pass_defense_rank",
            "rush_defense_rank",
            "scoring_defense_rank"
        ]),
        left_on="opponent",
        right_on="team"
    )
    
    # Calculate opponent difficulty multiplier (1.0-1.5 scale)
    # Lower rank (better defense) = higher multiplier
    return df_with_opponent.with_columns([
        (1 + (33 - pl.col("pass_defense_rank")) / 64).alias("pass_defense_multiplier"),
        (1 + (33 - pl.col("rush_defense_rank")) / 64).alias("rush_defense_multiplier"),
        (1 + (33 - pl.col("scoring_defense_rank")) / 64).alias("scoring_multiplier")
    ])

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
                    if m in ['receiving_yards', 'receiving_tds', 'receptions', 'targets']:
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
                    if m in ['receiving_yards', 'receiving_tds', 'receptions', 'targets']:
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
    
    # Get unique teams
    teams = sorted(team_stats['team'].unique().to_list())
    
    markdown = f"# Season Summary - {year}\n\n"
    
    # Create a table for each skill position metric
    for metric, metric_name in SKILL_POSITION_METRICS.items():
            
        table = PrettyTable()
        table.title = f"{metric_name} Share Leaders"
        table.field_names = ["Team", "Player", "Position", "Season Share (%)"]
        table.align = "l"
        table.float_format = '.1'
        table.set_style(TableStyle.MARKDOWN)
        
        for team in teams:
            try:
                # Get all weeks for this team
                team_season = team_stats.filter(pl.col('team') == team)
                player_season = player_stats.filter(pl.col('team') == team)
                
                # Calculate week-by-week shares with situational adjustments
                all_shares = []
                for week in team_season['week'].unique().to_list():
                    team_week = team_season.filter(pl.col('week') == week)
                    player_week = player_season.filter(pl.col('week') == week)
                    
                    week_shares = calculate_offensive_shares(team_week, player_week, metric, year=year, week=week, team=team)
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
    
    # Check if pressure data is available (2024 and earlier have ~96% coverage)
    has_pressure_data = pbp_data['was_pressure'].null_count() < len(pbp_data) * 0.5
    
    weights = COMBINED_METRICS['qb_contribution']['weights']
    contribution = 0.0
    
    # Get all plays for this QB
    qb_plays = pbp_data.filter(pl.col('passer_player_name') == qb_name)
    
    if len(qb_plays) == 0:
        return 0.0
    
    # === POSITIVE CONTRIBUTIONS ===
    
    # Passing yards - with pressure bonus if available
    if has_pressure_data:
        # Clean pocket yards
        clean_yards = qb_plays.filter(
            (pl.col('passing_yards').is_not_null()) & 
            ((pl.col('was_pressure') == 0) | (pl.col('was_pressure').is_null()))
        )['passing_yards'].sum()
        
        # Pressured yards (1.2x bonus)
        pressure_yards = qb_plays.filter(
            (pl.col('passing_yards').is_not_null()) & 
            (pl.col('was_pressure') == 1)
        )['passing_yards'].sum()
        
        contribution += (clean_yards or 0) * weights['passing_yards']
        contribution += (pressure_yards or 0) * weights['passing_yards'] * 1.2
    else:
        # No pressure data - use standard calculation
        total_yards = qb_plays.filter(pl.col('passing_yards').is_not_null())['passing_yards'].sum()
        contribution += (total_yards or 0) * weights['passing_yards']
    
    # Passing TDs
    passing_tds = qb_plays.filter(pl.col('pass_touchdown') == 1).height
    contribution += passing_tds * weights['passing_tds']
    
    # Completions - with pressure bonus if available
    if has_pressure_data:
        # Clean completions
        clean_completions = qb_plays.filter(
            (pl.col('complete_pass') == 1) & 
            ((pl.col('was_pressure') == 0) | (pl.col('was_pressure').is_null()))
        ).height
        
        # Pressured completions (1.4x bonus)
        pressure_completions = qb_plays.filter(
            (pl.col('complete_pass') == 1) & 
            (pl.col('was_pressure') == 1)
        ).height
        
        contribution += clean_completions * weights['completions']
        contribution += pressure_completions * weights['completions'] * 1.4
    else:
        # No pressure data - use standard calculation
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
    """Generate a separate QB rankings table with normalized 0-100 scores."""
    logger.info(f"Generating QB rankings for {year}")
    
    # Load QB stats
    qb_stats = load_position_weekly_stats(year, 'QB')
    if qb_stats is None:
        return "No QB data available."
    
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
    qb_contributions = []
    
    logger.info(f"Calculating QB contributions from play-by-play data with contextual penalties and pressure bonuses...")
    
    for qb_name in qb_stats['player_name'].unique().to_list():
        qb_data = qb_stats.filter(pl.col('player_name') == qb_name)
        
        # Calculate contribution from PBP data (includes penalties and pressure bonuses)
        contribution = calculate_qb_contribution_from_pbp(year, qb_name)
        
        games_played = qb_data.height
        if games_played > 0:
            qb_contributions.append({
                'player_name': qb_name,
                'team': qb_data['team'][0],
                'contribution': contribution,
                'games': games_played,
                'avg_per_game': contribution / games_played
            })
    
    if not qb_contributions:
        return "No QB contributions calculated."
    
    # Create DataFrame and apply z-score normalization
    qb_df = pl.DataFrame(qb_contributions)
    
    # Calculate mean and standard deviation
    mean_contribution = qb_df['contribution'].mean()
    std_contribution = qb_df['contribution'].std()
    
    # Calculate team-level max games (to penalize QBs who missed games their team played)
    # Group by team to count how many distinct weeks (games) each team has played
    team_games = qb_stats.group_by('team').agg([
        pl.col('week').n_unique().alias('team_max_games')
    ])
    
    # Join team max games back to qb_df
    qb_df = qb_df.join(team_games, on='team', how='left')
    
    logger.info(f"Team games played: {team_games}")
    
    # Z-score normalization: normalized = 50 + (z_score * 17.5)
    # Hybrid scoring: contribution * (qb_games / team_games)^0.4
    # This penalizes QBs who missed games their team played, but not for bye weeks
    qb_df = qb_df.with_columns([
        (50 + ((pl.col('contribution') - mean_contribution) / std_contribution) * 17.5).alias('normalized_score'),
        (pl.col('contribution') * (pl.col('games') / pl.col('team_max_games')).pow(0.4)).alias('hybrid_score')
    ])
    
    # Calculate z-score for hybrid scores to normalize those too
    mean_hybrid = qb_df['hybrid_score'].mean()
    std_hybrid = qb_df['hybrid_score'].std()
    
    qb_df = qb_df.with_columns([
        (50 + ((pl.col('hybrid_score') - mean_hybrid) / std_hybrid) * 17.5).alias('normalized_hybrid')
    ]).sort('normalized_hybrid', descending=True)
    
    # Create table
    markdown = f"# QB Rankings - {year}\n\n"
    table = PrettyTable()
    table.field_names = ["Rank", "QB", "Team", "Games", "Contribution", "Avg/Game", "Normalized"]
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)
    
    for rank, row in enumerate(qb_df.iter_rows(named=True), 1):
        table.add_row([
            rank,
            row['player_name'],
            row['team'],
            row['games'],
            f"{row['contribution']:.2f}",
            f"{row['avg_per_game']:.2f}",
            f"{row['normalized_hybrid']:.2f}"
        ])
    
    markdown += table.get_string() + "\n\n"
    return markdown


def generate_rb_rankings(year: int) -> str:
    """Generate comprehensive RB rankings showing all qualified players."""
    logger.info(f"Generating RB rankings for {year}")
    
    # Load RB stats
    rb_stats = load_position_weekly_stats(year, 'RB')
    if rb_stats is None:
        return "No RB data available."
    
    # Apply minimum activity threshold: 8 carries per game
    rb_games = rb_stats.group_by(['player_id', 'player_name']).agg([
        pl.col('week').count().alias('games'),
        pl.col('carries').sum().alias('total_carries')
    ])
    
    rb_games = rb_games.with_columns([
        (pl.col('total_carries') / pl.col('games')).alias('carries_per_game')
    ])
    
    # Filter to qualified RBs (8+ carries per game)
    qualified_rbs = rb_games.filter(pl.col('carries_per_game') >= 8.0)
    rb_stats = rb_stats.join(
        qualified_rbs.select(['player_id', 'player_name']), 
        on=['player_id', 'player_name'], 
        how='inner'
    )
    
    if len(rb_stats) == 0:
        return "No qualified RBs (8+ carries/game) for this season."
    
    logger.info(f"Filtered to {len(qualified_rbs)} qualified RBs (8+ carries/game)")
    
    # Calculate RB contribution scores
    rb_contributions = []
    
    # Group by player_id to avoid combining players with same name
    for player_id in rb_stats['player_id'].unique().to_list():
        rb_data = rb_stats.filter(pl.col('player_id') == player_id)
        rb_name = rb_data['player_name'][0]
        
        # Calculate weighted contribution using overall_contribution metric
        weights = COMBINED_METRICS['overall_contribution']['weights']
        contribution = 0
        for metric, weight in weights.items():
            if metric in rb_data.columns:
                contribution += rb_data[metric].sum() * weight
        
        games_played = rb_data.height
        if games_played > 0:
            # Calculate difficulty multiplier if available
            difficulty = None
            if 'defenders_in_box_multiplier' in rb_data.columns:
                difficulty = rb_data['defenders_in_box_multiplier'].mean()
            
            # Calculate adjusted score
            adjusted_contribution = contribution * difficulty if difficulty else contribution
            
            # Calculate typical game (25th/75th percentile average)
            game_contributions = []
            for week_data in rb_data.iter_rows(named=True):
                week_contrib = 0
                for metric, weight in weights.items():
                    if metric in week_data:
                        week_contrib += week_data[metric] * weight
                game_contributions.append(week_contrib)
            
            game_contributions.sort()
            if len(game_contributions) >= 4:
                q1_idx = len(game_contributions) // 4
                q3_idx = 3 * len(game_contributions) // 4
                typical = (game_contributions[q1_idx] + game_contributions[q3_idx]) / 2
            else:
                typical = sum(game_contributions) / len(game_contributions)
            
            # Calculate consistency (±5% threshold)
            below_avg = sum(1 for g in game_contributions if g < typical * 0.95)
            at_avg = sum(1 for g in game_contributions if typical * 0.95 <= g <= typical * 1.05)
            above_avg = sum(1 for g in game_contributions if g > typical * 1.05)
            consistency = f"{below_avg}/{at_avg}/{above_avg}"
            
            # Calculate trend (compare first half vs second half of games)
            # But only if player has been active recently (played in last 2 weeks)
            max_week_in_season = rb_stats['week'].max()
            last_week_played = rb_data['week'].max()
            weeks_since_last_game = max_week_in_season - last_week_played
            
            if weeks_since_last_game > 2:
                # Player hasn't played recently - trend is stale/inactive
                trend_str = "INACTIVE"
            elif len(game_contributions) >= 6:
                # For 6+ games with recent activity: compare first half vs second half
                midpoint = len(game_contributions) // 2
                first_half = sum(game_contributions[:midpoint]) / midpoint
                second_half = sum(game_contributions[midpoint:]) / (len(game_contributions) - midpoint)
                trend = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                trend_str = f"{trend:+.1f}%"
            else:
                trend_str = "N/A"
            
            rb_contributions.append({
                'player_name': rb_name,
                'team': rb_data['team'][0],
                'games': games_played,
                'raw_score': contribution,
                'adjusted_score': adjusted_contribution,
                'difficulty': difficulty if difficulty else 1.0,
                'avg_per_game': contribution / games_played,
                'typical': typical,
                'consistency': consistency,
                'trend': trend_str
            })
    
    if not rb_contributions:
        return "No RB contributions calculated."
    
    # Create DataFrame and sort by adjusted score
    rb_df = pl.DataFrame(rb_contributions).sort('adjusted_score', descending=True)
    
    # Create table
    markdown = f"# RB Rankings - {year}\n\n"
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Games", "Raw", "Adjusted", "Difficulty", "Avg/Game", "Typical", "Consistency", "Trend"]
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)
    
    for rank, row in enumerate(rb_df.iter_rows(named=True), 1):
        table.add_row([
            rank,
            row['player_name'],
            row['team'],
            row['games'],
            f"{row['raw_score']:.2f}",
            f"{row['adjusted_score']:.2f}",
            f"{row['difficulty']:.3f}",
            f"{row['avg_per_game']:.2f}",
            f"{row['typical']:.2f}",
            row['consistency'],
            row['trend']
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
    
    # Calculate WR contribution scores
    wr_contributions = []
    
    # Group by player_id to avoid combining players with same name
    for player_id in wr_stats['player_id'].unique().to_list():
        wr_data = wr_stats.filter(pl.col('player_id') == player_id)
        wr_name = wr_data['player_name'][0]
        
        # Calculate weighted contribution using overall_contribution metric
        weights = COMBINED_METRICS['overall_contribution']['weights']
        contribution = 0
        for metric, weight in weights.items():
            if metric in wr_data.columns:
                contribution += wr_data[metric].sum() * weight
        
        games_played = wr_data.height
        if games_played > 0:
            # Calculate difficulty multiplier if available
            difficulty = None
            if 'coverage_multiplier' in wr_data.columns:
                difficulty = wr_data['coverage_multiplier'].mean()
            
            # Calculate adjusted score
            adjusted_contribution = contribution * difficulty if difficulty else contribution
            
            # Calculate typical game
            game_contributions = []
            for week_data in wr_data.iter_rows(named=True):
                week_contrib = 0
                for metric, weight in weights.items():
                    if metric in week_data:
                        week_contrib += week_data[metric] * weight
                game_contributions.append(week_contrib)
            
            game_contributions.sort()
            if len(game_contributions) >= 4:
                q1_idx = len(game_contributions) // 4
                q3_idx = 3 * len(game_contributions) // 4
                typical = (game_contributions[q1_idx] + game_contributions[q3_idx]) / 2
            else:
                typical = sum(game_contributions) / len(game_contributions)
            
            # Calculate consistency (±5% threshold)
            below_avg = sum(1 for g in game_contributions if g < typical * 0.95)
            at_avg = sum(1 for g in game_contributions if typical * 0.95 <= g <= typical * 1.05)
            above_avg = sum(1 for g in game_contributions if g > typical * 1.05)
            consistency = f"{below_avg}/{at_avg}/{above_avg}"
            
            # Calculate trend (compare first half vs second half of games)
            # But only if player has been active recently (played in last 2 weeks)
            max_week_in_season = wr_stats['week'].max()
            last_week_played = wr_data['week'].max()
            weeks_since_last_game = max_week_in_season - last_week_played
            
            if weeks_since_last_game > 2:
                # Player hasn't played recently - trend is stale/inactive
                trend_str = "INACTIVE"
            elif len(game_contributions) >= 6:
                # For 6+ games with recent activity: compare first half vs second half
                midpoint = len(game_contributions) // 2
                first_half = sum(game_contributions[:midpoint]) / midpoint
                second_half = sum(game_contributions[midpoint:]) / (len(game_contributions) - midpoint)
                trend = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                trend_str = f"{trend:+.1f}%"
            else:
                trend_str = "N/A"
            
            wr_contributions.append({
                'player_name': wr_name,
                'team': wr_data['team'][0],
                'games': games_played,
                'raw_score': contribution,
                'adjusted_score': adjusted_contribution,
                'difficulty': difficulty if difficulty else 1.0,
                'avg_per_game': contribution / games_played,
                'typical': typical,
                'consistency': consistency,
                'trend': trend_str
            })
    
    if not wr_contributions:
        return "No WR contributions calculated."
    
    # Create DataFrame and sort by adjusted score
    wr_df = pl.DataFrame(wr_contributions).sort('adjusted_score', descending=True)
    
    # Create table
    markdown = f"# WR Rankings - {year}\n\n"
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Games", "Raw", "Adjusted", "Difficulty", "Avg/Game", "Typical", "Consistency", "Trend"]
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)
    
    for rank, row in enumerate(wr_df.iter_rows(named=True), 1):
        table.add_row([
            rank,
            row['player_name'],
            row['team'],
            row['games'],
            f"{row['raw_score']:.2f}",
            f"{row['adjusted_score']:.2f}",
            f"{row['difficulty']:.3f}",
            f"{row['avg_per_game']:.2f}",
            f"{row['typical']:.2f}",
            row['consistency'],
            row['trend']
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
    
    # Calculate TE contribution scores
    te_contributions = []
    
    # Group by player_id to avoid combining players with same name
    for player_id in te_stats['player_id'].unique().to_list():
        te_data = te_stats.filter(pl.col('player_id') == player_id)
        te_name = te_data['player_name'][0]
        
        # Calculate weighted contribution using overall_contribution metric
        weights = COMBINED_METRICS['overall_contribution']['weights']
        contribution = 0
        for metric, weight in weights.items():
            if metric in te_data.columns:
                contribution += te_data[metric].sum() * weight
        
        games_played = te_data.height
        if games_played > 0:
            # Calculate difficulty multiplier if available
            difficulty = None
            if 'coverage_multiplier' in te_data.columns:
                difficulty = te_data['coverage_multiplier'].mean()
            
            # Calculate adjusted score
            adjusted_contribution = contribution * difficulty if difficulty else contribution
            
            # Calculate typical game
            game_contributions = []
            for week_data in te_data.iter_rows(named=True):
                week_contrib = 0
                for metric, weight in weights.items():
                    if metric in week_data:
                        week_contrib += week_data[metric] * weight
                game_contributions.append(week_contrib)
            
            game_contributions.sort()
            if len(game_contributions) >= 4:
                q1_idx = len(game_contributions) // 4
                q3_idx = 3 * len(game_contributions) // 4
                typical = (game_contributions[q1_idx] + game_contributions[q3_idx]) / 2
            else:
                typical = sum(game_contributions) / len(game_contributions)
            
            # Calculate consistency (±5% threshold)
            below_avg = sum(1 for g in game_contributions if g < typical * 0.95)
            at_avg = sum(1 for g in game_contributions if typical * 0.95 <= g <= typical * 1.05)
            above_avg = sum(1 for g in game_contributions if g > typical * 1.05)
            consistency = f"{below_avg}/{at_avg}/{above_avg}"
            
            # Calculate trend (compare first half vs second half of games)
            # But only if player has been active recently (played in last 2 weeks)
            max_week_in_season = te_stats['week'].max()
            last_week_played = te_data['week'].max()
            weeks_since_last_game = max_week_in_season - last_week_played
            
            if weeks_since_last_game > 2:
                # Player hasn't played recently - trend is stale/inactive
                trend_str = "INACTIVE"
            elif len(game_contributions) >= 6:
                # For 6+ games with recent activity: compare first half vs second half
                midpoint = len(game_contributions) // 2
                first_half = sum(game_contributions[:midpoint]) / midpoint
                second_half = sum(game_contributions[midpoint:]) / (len(game_contributions) - midpoint)
                trend = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                trend_str = f"{trend:+.1f}%"
            else:
                trend_str = "N/A"
            
            te_contributions.append({
                'player_name': te_name,
                'team': te_data['team'][0],
                'games': games_played,
                'raw_score': contribution,
                'adjusted_score': adjusted_contribution,
                'difficulty': difficulty if difficulty else 1.0,
                'avg_per_game': contribution / games_played,
                'typical': typical,
                'consistency': consistency,
                'trend': trend_str
            })
    
    if not te_contributions:
        return "No TE contributions calculated."
    
    # Create DataFrame and sort by adjusted score
    te_df = pl.DataFrame(te_contributions).sort('adjusted_score', descending=True)
    
    # Create table
    markdown = f"# TE Rankings - {year}\n\n"
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Games", "Raw", "Adjusted", "Difficulty", "Avg/Game", "Typical", "Consistency", "Trend"]
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)
    
    for rank, row in enumerate(te_df.iter_rows(named=True), 1):
        table.add_row([
            rank,
            row['player_name'],
            row['team'],
            row['games'],
            f"{row['raw_score']:.2f}",
            f"{row['adjusted_score']:.2f}",
            f"{row['difficulty']:.3f}",
            f"{row['avg_per_game']:.2f}",
            f"{row['typical']:.2f}",
            row['consistency'],
            row['trend']
        ])
    
    markdown += table.get_string() + "\n\n"
    return markdown


def apply_phase4_adjustments(contributions: pl.DataFrame, year: int) -> pl.DataFrame:
    """
    Apply Phase 4 player-level adjustments (catch rate and blocking quality).
    
    These adjustments are calculated once per player after season aggregation.
    Unlike Phase 1-3 multipliers (applied per-play in cache), these require
    full season context to calculate properly.
    
    Args:
        contributions: DataFrame with player weekly contributions
        year: Season year to load PBP data from
        
    Returns:
        DataFrame with Phase 4 adjustments applied to player_overall_contribution
    """
    logger.info(f"Applying Phase 4 adjustments (catch rate + blocking quality) for {year}")
    
    # Load full season PBP data for catch rate and blocking quality calculations
    try:
        pbp_data = pbp_processor.load_pbp_data(year)
        if pbp_data is None or pbp_data.height == 0:
            logger.warning(f"No PBP data available for {year}, skipping Phase 4 adjustments")
            return contributions
    except Exception as e:
        logger.error(f"Error loading PBP data for Phase 4 adjustments: {str(e)}")
        return contributions
    
    # Calculate adjustments for each unique player
    adjustments = []
    unique_players = contributions.select(['player_id', 'player_name', 'team', 'position']).unique()
    
    for player_row in unique_players.iter_rows(named=True):
        player_id = player_row['player_id']
        player_name = player_row['player_name']
        player_team = player_row['team']
        position = player_row['position']
        
        catch_rate_mult = 1.0
        blocking_mult = 1.0
        
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
        
        adjustments.append({
            'player_id': player_id,
            'player_name': player_name,
            'team': player_team,
            'catch_rate_adjustment': catch_rate_mult,
            'blocking_adjustment': blocking_mult
        })
    
    # Create DataFrame and join to contributions
    adj_df = pl.DataFrame(adjustments)
    contributions = contributions.join(adj_df, on=['player_id', 'player_name', 'team'], how='left')
    
    # Fill missing adjustments with 1.0 (neutral)
    contributions = contributions.with_columns([
        pl.col('catch_rate_adjustment').fill_null(1.0),
        pl.col('blocking_adjustment').fill_null(1.0)
    ])
    
    # Apply combined Phase 4 multiplier to player_overall_contribution
    contributions = contributions.with_columns([
        (pl.col('player_overall_contribution') * 
         pl.col('catch_rate_adjustment') * 
         pl.col('blocking_adjustment')).alias('player_overall_contribution')
    ])
    
    logger.info(f"Phase 4 adjustments applied to {len(unique_players)} players")
    
    return contributions


def apply_phase5_adjustments(contributions: pl.DataFrame) -> pl.DataFrame:
    """
    Apply Phase 5 adjustments (talent context + sample size dampening).
    
    Two-pass system:
    1. Use baseline scores from contributions to calculate teammate quality
    2. Apply talent adjustment multiplier
    3. Apply sample size dampening based on games played
    
    Args:
        contributions: DataFrame with player weekly contributions including baseline scores
        
    Returns:
        DataFrame with Phase 5 adjustments applied
    """
    logger.info("Applying Phase 5 adjustments (talent context + sample size dampening)")
    
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
    player_final = contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').mean().alias('score_before_dampening'),
        pl.col('week').count().alias('games_played')
    ])
    
    # Calculate dampened scores
    dampened_scores = []
    for row in player_final.iter_rows(named=True):
        score = row['score_before_dampening']
        games = row['games_played']
        
        dampened = context_adj.apply_sample_size_dampening(score, games, full_season_games=17)
        
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
    raw_contributions = calculate_offensive_shares(team_stats, player_stats, 'overall_contribution', year=None)
    
    # Calculate ADJUSTED scores (with situational adjustments from PBP data)
    contributions = calculate_offensive_shares(team_stats, player_stats, 'overall_contribution', year=year)
    
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
    contributions = apply_phase5_adjustments(contributions)
    
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
    overview_table.field_names = ["Rank", "Player", "Team", "Position (Pos. Rank)", "Adjusted Score", 
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
            pl.col('player_overall_contribution').mean().alias('adjusted_score'),
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
        pl.col('player_overall_contribution').mean().alias('adjusted_score'),
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
            pl.col('player_overall_contribution').mean().alias('raw_score')
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
        raw_score = row['raw_score']
        adj_score_per_game = row['adjusted_score']  # Phase 5 returns per-game dampened score
        games_played = row['games_played']
        adj_score_total = adj_score_per_game * games_played  # Calculate season total for display
        avg_per_game = adj_score_per_game  # Already per-game from Phase 5
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
        overview_table.add_row([rank, player_name, team, position_display, f"{adj_score_total:.2f}", 
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
                pl.col('player_overall_contribution').mean().alias('adjusted_score'),
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
            adj_score_per_game = row['adjusted_score']  # Phase 5 returns per-game dampened score
            games_played = row['games_played']
            adj_score_total = adj_score_per_game * games_played  # Calculate season total for display
            avg_per_game = adj_score_per_game  # Already per-game from Phase 5
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
    """Check all years for missing difficulty columns and rebuild caches as needed.
    
    This runs before main processing to ensure all caches have required columns.
    Rebuilds happen in parallel for speed.
    
    Args:
        years: List of years to check
        parallel: Whether to rebuild in parallel (default True)
    """
    logger.info("Checking cache completeness for all years...")
    
    required_cols = ['defenders_in_box', 'defense_coverage_type', 
                     'defenders_in_box_multiplier', 'coverage_multiplier']
    
    years_needing_rebuild = []
    
    # Check each year's cache
    for year in years:
        pbp_path = Path("cache/pbp") / f"pbp_{year}.parquet"
        if not pbp_path.exists():
            logger.info(f"Cache missing for {year}, will rebuild")
            years_needing_rebuild.append(year)
        else:
            try:
                pbp_data = pl.read_parquet(pbp_path)
                if not all(col in pbp_data.columns for col in required_cols):
                    logger.info(f"Cache for {year} missing difficulty columns, will rebuild")
                    years_needing_rebuild.append(year)
            except Exception as e:
                logger.warning(f"Error reading cache for {year}: {e}, will rebuild")
                years_needing_rebuild.append(year)
    
    if not years_needing_rebuild:
        logger.info("All caches up to date!")
        return
    
    logger.info(f"Rebuilding caches for {len(years_needing_rebuild)} years: {years_needing_rebuild}")
    
    # Rebuild caches
    if parallel and len(years_needing_rebuild) > 1:
        from concurrent.futures import ThreadPoolExecutor
        logger.info(f"Rebuilding {len(years_needing_rebuild)} caches in parallel...")
        with ThreadPoolExecutor(max_workers=min(len(years_needing_rebuild), 8)) as executor:
            futures = {executor.submit(build_cache, year, True): year for year in years_needing_rebuild}
            for future in concurrent.futures.as_completed(futures):
                year = futures[future]
                try:
                    success = future.result()
                    if success:
                        logger.info(f"✓ Cache rebuilt for {year}")
                    else:
                        logger.warning(f"✗ Cache rebuild failed for {year}")
                except Exception as e:
                    logger.error(f"✗ Cache rebuild error for {year}: {e}")
    else:
        # Sequential rebuild
        for year in years_needing_rebuild:
            try:
                logger.info(f"Rebuilding cache for {year}...")
                success = build_cache(year, force=True)
                if success:
                    logger.info(f"✓ Cache rebuilt for {year}")
                else:
                    logger.warning(f"✗ Cache rebuild failed for {year}")
            except Exception as e:
                logger.error(f"✗ Cache rebuild error for {year}: {e}")
    
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
