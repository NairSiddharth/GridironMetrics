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
import concurrent.futures
import argparse

logger = get_logger(__name__)

# Initialize our processors
metrics_calculator = OffensiveMetricsCalculator()
pbp_processor = PlayByPlayProcessor()

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
                        pl.col("down_multiplier").mean().alias("down_multiplier")
                    ])
                    multipliers.append({
                        'team': t,
                        'week': w,
                        'redzone_multiplier': float(gm[0, 'redzone_multiplier']),
                        'game_state_multiplier': float(gm[0, 'game_state_multiplier']),
                        'time_multiplier': float(gm[0, 'time_multiplier']),
                        'down_multiplier': float(gm[0, 'down_multiplier'])
                    })
                else:
                    multipliers.append({
                        'team': t, 'week': w,
                        'redzone_multiplier': 1.0, 'game_state_multiplier': 1.0,
                        'time_multiplier': 1.0, 'down_multiplier': 1.0
                    })
            except Exception:
                multipliers.append({
                    'team': t, 'week': w,
                    'redzone_multiplier': 1.0, 'game_state_multiplier': 1.0,
                    'time_multiplier': 1.0, 'down_multiplier': 1.0
                })

        if len(multipliers) > 0:
            mult_df = pl.DataFrame(multipliers)
            # join multipliers onto team_stats and player_stats
            team_stats = team_stats.join(mult_df, on=['team', 'week'], how='left')
            player_stats = player_stats.join(mult_df, on=['team', 'week'], how='left')
        else:
            team_stats = team_stats.with_columns([
                pl.lit(1.0).alias("redzone_multiplier"), pl.lit(1.0).alias("game_state_multiplier"),
                pl.lit(1.0).alias("time_multiplier"), pl.lit(1.0).alias("down_multiplier")
            ])
            player_stats = player_stats.with_columns([
                pl.lit(1.0).alias("redzone_multiplier"), pl.lit(1.0).alias("game_state_multiplier"),
                pl.lit(1.0).alias("time_multiplier"), pl.lit(1.0).alias("down_multiplier")
            ])
    else:
        # No year provided -> raw values: set multipliers to 1.0
        team_stats = team_stats.with_columns([
            pl.lit(1.0).alias("redzone_multiplier"), pl.lit(1.0).alias("game_state_multiplier"),
            pl.lit(1.0).alias("time_multiplier"), pl.lit(1.0).alias("down_multiplier")
        ])
        player_stats = player_stats.with_columns([
            pl.lit(1.0).alias("redzone_multiplier"), pl.lit(1.0).alias("game_state_multiplier"),
            pl.lit(1.0).alias("time_multiplier"), pl.lit(1.0).alias("down_multiplier")
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


def generate_qb_rankings(year: int) -> str:
    """Generate a separate QB rankings table with normalized 0-100 scores."""
    logger.info(f"Generating QB rankings for {year}")
    
    # Load QB stats
    qb_stats = load_position_weekly_stats(year, 'QB')
    if qb_stats is None:
        return "No QB data available."
    
    # Calculate QB contribution scores using the qb_contribution metric
    qb_contributions = []
    
    for qb_name in qb_stats['player_name'].unique().to_list():
        qb_data = qb_stats.filter(pl.col('player_name') == qb_name)
        
        # Calculate weighted contribution
        weights = COMBINED_METRICS['qb_contribution']['weights']
        contribution = 0
        for metric, weight in weights.items():
            if metric in qb_data.columns:
                contribution += qb_data[metric].sum() * weight
        
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
    
    # Calculate max games for hybrid scoring
    max_games = qb_df['games'].max()
    
    # Z-score normalization: normalized = 50 + (z_score * 10)
    # Hybrid scoring: adjusted_score * (games_played / max_games)^0.4
    qb_df = qb_df.with_columns([
        (50 + ((pl.col('contribution') - mean_contribution) / std_contribution) * 10).alias('normalized_score'),
        (pl.col('contribution') * (pl.col('games') / max_games).pow(0.4)).alias('hybrid_score')
    ])
    
    # Calculate z-score for hybrid scores to normalize those too
    mean_hybrid = qb_df['hybrid_score'].mean()
    std_hybrid = qb_df['hybrid_score'].std()
    
    qb_df = qb_df.with_columns([
        (50 + ((pl.col('hybrid_score') - mean_hybrid) / std_hybrid) * 10).alias('normalized_hybrid')
    ]).sort('normalized_hybrid', descending=True)
    
    # Create table
    markdown = f"# QB Rankings - {year}\n\n"
    table = PrettyTable()
    table.field_names = ["Rank", "QB", "Team", "Games", "Avg/Game", "Normalized"]
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)
    
    for rank, row in enumerate(qb_df.iter_rows(named=True), 1):
        table.add_row([
            rank,
            row['player_name'],
            row['team'],
            row['games'],
            f"{row['avg_per_game']:.2f}",
            f"{row['normalized_hybrid']:.2f}"
        ])
    
    markdown += table.get_string() + "\n\n"
    return markdown

               
        
def generate_top_contributors(year: int) -> str:
    """Generate a table of the top 10 offensive contributors for the season."""
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
    
    # First, calculate positional rankings (use raw scores for ranking)
    # Rank by (player_id, player_name, team) to handle players with same name on different teams
    position_rankings = {}
    for pos in SKILL_POSITIONS:
        pos_stats = raw_season_contributions.filter(
            pl.col('position') == pos
        ).sort('raw_score', descending=True)
        
        position_rankings[pos] = {
            (row['player_id'], row['player_name'], row['team']): rank + 1 
            for rank, row in enumerate(pos_stats.iter_rows(named=True))
        }
    
    
    # Add position to contributions for later use
    contributions = contributions.join(player_positions, on=['player_id', 'player_name'], how='left')
    
    markdown = f"# Top Contributors - {year}\n\n"
    markdown += "## Overall Rankings\n\n"
    
    # Create PrettyTable for top contributors
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Position (Pos. Rank)", "Raw Score", "Adjusted Score", 
                        "Games", "Avg/Game", "Peak Performance", "Trend", "Notable Games"]
    table.align = "l"  # Left align text
    table.float_format = '.2'  # Two decimal places for floats
    table.set_style(TableStyle.MARKDOWN)
    
    # Get top 10 by adjusted contribution
    top_contributors = (
        contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('adjusted_score'),
            pl.col('player_overall_contribution').max().alias('peak_score'),
            pl.col('week').count().alias('games_played')
        ])
        .sort('adjusted_score', descending=True)
        .head(10)
    )
    
    # Add raw scores from raw_contributions
    raw_scores = (
        raw_contributions.group_by(['player_id', 'player_name', 'team'])
        .agg([
            pl.col('player_overall_contribution').mean().alias('raw_score')
        ])
    )
    top_contributors = top_contributors.join(raw_scores, on=['player_id', 'player_name', 'team'], how='left')
    
    # Get notable games (weeks where player had >150% of their average contribution)
    # Need to get opponent info from player_stats since it's not in contributions
    notable_games = {}
    for row in top_contributors.iter_rows(named=True):
        player_id = row['player_id']
        player_name = row['player_name']
        team_name = row['team']
        avg_score = row['adjusted_score']
        
        # Get player weeks with opponent information from player_stats
        player_weeks = player_stats.filter(
            (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name) & (pl.col('team') == team_name)
        ).select(['week', 'opponent_team']).unique()
        
        # Get contribution scores from contributions
        player_contrib = contributions.filter(
            (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name) & (pl.col('team') == team_name)
        ).select(['week', 'player_overall_contribution'])
        
        # Join to get both opponent and contribution
        player_full = player_contrib.join(player_weeks, on='week', how='left')
        
        high_games = player_full.filter(
            pl.col('player_overall_contribution') > avg_score * 1.5
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
                        "Games", "Avg/Game", "Peak Performance", "Trend", "Notable Games (>150% Avg)"]
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
        adj_score = row['adjusted_score']
        games_played = row['games_played']
        avg_per_game = adj_score / games_played if games_played > 0 else 0
        peak = f"Peak: {row['peak_score']:.2f}"
        
        # Calculate trend (simple: compare first half to second half of season)
        player_games = contributions.filter((pl.col('player_id') == player_id) & (pl.col('player_name') == player_name)).sort('week')
        if len(player_games) >= 4:
            mid_point = len(player_games) // 2
            first_half_avg = player_games.head(mid_point)['player_overall_contribution'].mean()
            second_half_avg = player_games.tail(len(player_games) - mid_point)['player_overall_contribution'].mean()
            if second_half_avg > first_half_avg * 1.1:
                trend = "Increasing"
            elif second_half_avg < first_half_avg * 0.9:
                trend = "Decreasing"
            else:
                trend = "Stable"
        else:
            trend = "Stable"
        
        games = notable_games.get(player_name, "")
        
        table.add_row([rank, player_name, team, position_display, f"{raw_score:.2f}", f"{adj_score:.2f}", 
                      games_played, f"{avg_per_game:.2f}", peak, trend, games])
    
    markdown += table.get_string() + "\n\n"
    
    # Add positional rankings for RB, WR, TE
    for pos in ['RB', 'WR', 'TE']:
        markdown += f"## {pos} Rankings\n\n"
        
        # Get top 10 players at this position
        pos_contributors = (
            contributions.filter(pl.col('position') == pos)
            .group_by(['player_id', 'player_name', 'team'])
            .agg([
                pl.col('player_overall_contribution').mean().alias('adjusted_score'),
                pl.col('player_overall_contribution').max().alias('peak_score'),
                pl.col('week').count().alias('games_played')
            ])
            .sort('adjusted_score', descending=True)
            .head(10)
        )
        
        # Add raw scores
        pos_contributors = pos_contributors.join(raw_scores, on=['player_id', 'player_name', 'team'], how='left')
        
        # Create table for this position
        pos_table = PrettyTable()
        pos_table.field_names = ["Rank", "Player", "Team", "Raw Score", "Adjusted Score", 
                                "Games", "Avg/Game", "Peak Performance", "Trend", "Notable Games (>150% Avg)"]
        pos_table.align = "l"
        pos_table.float_format = '.2'
        pos_table.set_style(TableStyle.MARKDOWN)
        
        for rank, row in enumerate(pos_contributors.iter_rows(named=True), 1):
            player_id = row['player_id']
            player_name = row['player_name']
            team = row['team']
            raw_score = row['raw_score']
            adj_score = row['adjusted_score']
            games_played = row['games_played']
            avg_per_game = adj_score / games_played if games_played > 0 else 0
            peak = f"Peak: {row['peak_score']:.2f}"
            
            # Calculate trend
            player_games = contributions.filter(
                (pl.col('player_id') == player_id) & (pl.col('player_name') == player_name) & (pl.col('position') == pos) & (pl.col('team') == team)
            ).sort('week')
            if len(player_games) >= 4:
                mid_point = len(player_games) // 2
                first_half_avg = player_games.head(mid_point)['player_overall_contribution'].mean()
                second_half_avg = player_games.tail(len(player_games) - mid_point)['player_overall_contribution'].mean()
                if second_half_avg > first_half_avg * 1.1:
                    trend = "Increasing"
                elif second_half_avg < first_half_avg * 0.9:
                    trend = "Decreasing"
                else:
                    trend = "Stable"
            else:
                trend = "Stable"
            
            # Get notable games for this player (reuse from notable_games dict or recalculate)
            # Use (player_id, player_name) key since that's unique
            games = notable_games.get(player_name, "")
            
            pos_table.add_row([rank, player_name, team, f"{raw_score:.2f}", f"{adj_score:.2f}", 
                              games_played, f"{avg_per_game:.2f}", peak, trend, games])
        
        markdown += pos_table.get_string() + "\n\n"
    
    return markdown

def process_year(year: int) -> tuple[bool, str, str, str, str]:
    """Process a single year's data with error recovery.
    
    Returns:
        tuple of (success, weekly_markdown, summary_markdown, top_contributors_markdown, qb_rankings_markdown)
    """
    try:
        logger.info(f"Processing year {year}")
        weekly_markdown = generate_weekly_tables(year)
        summary_markdown = generate_season_summary(year)
        top_contributors = generate_top_contributors(year)
        qb_rankings = generate_qb_rankings(year)
        return True, weekly_markdown, summary_markdown, top_contributors, qb_rankings
    except Exception as e:
        logger.error(f"Failed to process year {year}: {str(e)}")
        error_msg = f"Error processing {year}: {str(e)}"
        return False, error_msg, error_msg, error_msg, error_msg

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
    results = {}
    
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(years), 8)) as executor:
            future_to_year = {executor.submit(process_year, year): year for year in years}
            for future in concurrent.futures.as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    success, weekly, summary, top_contributors, qb_rankings = future.result()
                    results[year] = (success, weekly, summary, top_contributors, qb_rankings)
                except Exception as e:
                    logger.error(f"Year {year} failed with error: {str(e)}")
                    results[year] = (False, str(e), str(e), str(e), str(e))
    else:
        for year in years:
            success, weekly, summary, top_contributors, qb_rankings = process_year(year)
            results[year] = (success, weekly, summary, top_contributors, qb_rankings)
    
    # Save results and build index
    successful_years = []
    failed_years = []
    
    for year, (success, weekly, summary, top_contributors, qb_rankings) in results.items():
        output_dir = output_base / str(year)
        output_dir.mkdir(exist_ok=True)
        
        if success:
            successful_years.append(year)
            weekly_file = output_dir / "weekly_analysis.md"
            summary_file = output_dir / "season_summary.md"
            top_contributors_file = output_dir / "top_contributors.md"
            qb_rankings_file = output_dir / "qb_rankings.md"
            weekly_file.write_text(weekly)
            summary_file.write_text(summary)
            top_contributors_file.write_text(top_contributors)
            qb_rankings_file.write_text(qb_rankings)
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
            index_markdown += f"- [{year}](/{year}/)\n"
            index_markdown += f"  - [Weekly Analysis](/{year}/weekly_analysis.md)\n"
            index_markdown += f"  - [Season Summary](/{year}/season_summary.md)\n"
            index_markdown += f"  - [Top Contributors](/{year}/top_contributors.md)\n"
            index_markdown += f"  - [QB Rankings](/{year}/qb_rankings.md)\n\n"
    
    if failed_years:
        index_markdown += "## Failed Years\n\n"
        for year in sorted(failed_years):
            index_markdown += f"- [{year}](/{year}/error.md)\n\n"
    
    index_file = output_base / "index.md"
    index_file.write_text(index_markdown)
    logger.info(f"Saved multi-year index to {index_file}")
    
    if failed_years:
        logger.warning(f"Failed to process years: {failed_years}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NFL offensive share analysis")
    parser.add_argument("--start-year", type=int, help="Start year (default: START_YEAR from constants)")
    parser.add_argument("--end-year", type=int, help="End year (default: END_YEAR from constants)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    args = parser.parse_args()
    
    main(args.start_year, args.end_year, not args.no_parallel)
