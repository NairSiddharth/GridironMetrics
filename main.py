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

# Offensive metrics we want to track
OFFENSIVE_METRICS = {
    'passing_yards': 'Pass Yards',
    'rushing_yards': 'Rush Yards',
    'receiving_yards': 'Rec Yards',
    'passing_touchdowns': 'Pass TD',
    'rushing_touchdowns': 'Rush TD',
    'receiving_touchdowns': 'Rec TD',
    'receptions': 'Receptions',
    'targets': 'Targets',
    'rushing_attempts': 'Rush Att'
}

# Metric groupings for combined stats
COMBINED_METRICS = {
    'total_yards': {
        'metrics': ['receiving_yards', 'rushing_yards'],
        'display': 'Total Yards (Rush + Rec)'
    },
    'total_touchdowns': {
        'metrics': ['receiving_touchdowns', 'rushing_touchdowns'],
        'display': 'Total TDs (Rush + Rec)'
    },
    'overall_contribution': {
        'metrics': ['receiving_yards', 'rushing_yards', 'receiving_touchdowns', 'rushing_touchdowns', 'receptions', 'targets', 'rushing_attempts'],
        'display': 'Overall Offensive Contribution',
        'weights': {  # Weights based on EPA and WPA analysis
            'receiving_yards': 1.0,      # Base unit for comparison
            'rushing_yards': 1.0,        # Equal to receiving yards in value
            'receiving_touchdowns': 50.0, # Based on EPA conversion (~7 points / 0.14 points per yard)
            'rushing_touchdowns': 50.0,   # Equal to receiving TD in value
            'receptions': 8.0,           # Success rate and chain-moving value beyond yards
            'targets': 3.0,              # Opportunity cost and defensive attention value
            'rushing_attempts': 3.0       # Similar opportunity value to targets
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
                    df = pl.read_csv(file_path)
                    team_data.append(df)
        
        if not team_data:
            logger.error(f"No team data found for {year}")
            return None
            
        return pl.concat(team_data)
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
        return pl.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading {position} stats for {year}: {str(e)}")
        return None

def adjust_for_game_situation(df: pl.DataFrame) -> pl.DataFrame:
    """Apply situational adjustments to stats based on game context."""
    return df.with_columns([
        # Red zone multiplier (within 20 yard line)
        pl.when(pl.col("field_position") <= 20)
        .then(1.5)  # 50% bonus for red zone production
        .otherwise(1.0)
        .alias("redzone_multiplier"),
        
        # Score differential adjustment
        pl.when(pl.abs(pl.col("score_differential")) <= 7)
        .then(1.3)  # 30% bonus for close game production
        .when(pl.abs(pl.col("score_differential")) <= 14)
        .then(1.1)  # 10% bonus for moderately close games
        .otherwise(1.0)
        .alias("game_state_multiplier"),
        
        # Time remaining adjustment (4th quarter/OT in close games)
        pl.when((pl.col("quarter").is_in([4, 5])) & (pl.abs(pl.col("score_differential")) <= 8))
        .then(1.4)  # 40% bonus for crunch time
        .otherwise(1.0)
        .alias("time_multiplier"),
        
        # Down and distance context
        pl.when(pl.col("down") == 3)
        .then(1.25)  # 25% bonus for 3rd down conversions
        .when(pl.col("down") == 4)
        .then(1.5)   # 50% bonus for 4th down conversions
        .otherwise(1.0)
        .alias("down_multiplier")
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
                             opponent_stats: pl.DataFrame = None) -> pl.DataFrame:
    """Calculate what percentage of team total each player accounts for with situational and opponent adjustments."""
    # Apply situational adjustments first
    team_stats = adjust_for_game_situation(team_stats)
    player_stats = adjust_for_game_situation(player_stats)
    
    # Apply opponent adjustments if available
    if opponent_stats is not None:
        team_stats = adjust_for_opponent(team_stats, opponent_stats)
        player_stats = adjust_for_opponent(player_stats, opponent_stats)
    
    if metric in COMBINED_METRICS:
        # For combined metrics, sum the component metrics first
        combined_info = COMBINED_METRICS[metric]
        for df in [team_stats, player_stats]:
            if metric == 'overall_contribution':
                # For overall contribution, apply weights with situational and opponent adjustments
                weighted_cols = []
                max_possible_score = 0  # Calculate theoretical max for normalization
                for m in combined_info['metrics']:
                    base_weight = combined_info['weights'][m]
                    
                    # Apply situational multipliers
                    situational_weight = (
                        pl.col(m) * base_weight *
                        pl.col("redzone_multiplier") *
                        pl.col("game_state_multiplier") *
                        pl.col("time_multiplier") *
                        pl.col("down_multiplier")
                    )
                    
                    # Apply opponent adjustments if available
                    if any(col in df.columns for col in ["pass_defense_multiplier", "rush_defense_multiplier"]):
                        if m in ['receiving_yards', 'receiving_touchdowns', 'receptions', 'targets']:
                            situational_weight = situational_weight * pl.col("pass_defense_multiplier")
                        elif m in ['rushing_yards', 'rushing_touchdowns', 'rushing_attempts']:
                            situational_weight = situational_weight * pl.col("rush_defense_multiplier")
                    
                    weighted_cols.append(situational_weight)
                    
                    # Calculate max possible score with all bonuses
                    max_values = {
                        'receiving_yards': 400, 'rushing_yards': 400,
                        'receiving_touchdowns': 4, 'rushing_touchdowns': 4,
                        'receptions': 15, 'targets': 20, 'rushing_attempts': 20
                    }
                    # Maximum multiplier combination (all situational bonuses * max opponent adjustment)
                    max_multiplier = 1.5 * 1.3 * 1.4 * 1.5 * 1.5  # Product of max possible multipliers
                    max_possible_score += max_values[m] * base_weight * max_multiplier
                
                # Sum all weighted contributions and normalize to 0-100 scale
                df = df.with_columns([
                    (pl.fold(0, lambda acc, x: acc + x, weighted_cols) / max_possible_score * 100)
                    .alias(metric)
                ])
            else:
                # For simple combined metrics, just sum the components
                df = df.with_columns([
                    pl.fold(0, lambda acc, x: acc + x, [pl.col(m) for m in combined_info['metrics']])
                    .alias(metric)
                ])
    
    # Group by team and week to get team totals
    team_totals = team_stats.groupby(['team', 'week']).agg([
        pl.col(metric).sum().alias(f'team_{metric}')
    ])
    
    # Group by player, team, and week to get player totals
    player_totals = player_stats.groupby(['player_name', 'team', 'week']).agg([
        pl.col(metric).sum().alias(f'player_{metric}')
    ])
    
    # Join and calculate percentages
    return player_totals.join(
        team_totals,
        on=['team', 'week']
    ).with_columns([
        (pl.col(f'player_{metric}') / pl.col(f'team_{metric}') * 100).alias(f'{metric}_share')
    ])

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
            
            # Create a table for each offensive metric
            for metric, metric_name in OFFENSIVE_METRICS.items():
                try:
                    shares = calculate_offensive_shares(
                        team_stats.filter(pl.col('team') == team).filter(pl.col('week') == week),
                        player_stats.filter(pl.col('team') == team).filter(pl.col('week') == week),
                        metric
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
    
    # Create a table for each offensive metric
    for metric, metric_name in OFFENSIVE_METRICS.items():
        table = PrettyTable()
        table.title = f"{metric_name} Share Leaders"
        table.field_names = ["Team", "Player", "Position", "Season Share (%)"]
        table.align = "l"
        table.float_format = '.1'
        table.set_style(TableStyle.MARKDOWN)
        
        for team in teams:
            try:
                # Calculate season-long shares
                team_season = team_stats.filter(pl.col('team') == team)
                player_season = player_stats.filter(pl.col('team') == team)
                shares = calculate_offensive_shares(team_season, player_season, metric)
                
                if shares.height == 0:
                    continue
                
                # Get the top contributor
                top_player = shares.sort(f'{metric}_share', descending=True).head(1)
                if top_player.height > 0:
                    player_name = top_player['player_name'][0]
                    position = player_stats.filter(pl.col('player_name') == player_name)['position'].head(1)[0]
                    share = top_player[f'{metric}_share'][0]
                    table.add_row([team, player_name, position, share])
            except Exception as e:
                logger.error(f"Error generating summary for {team} {metric}: {str(e)}")
                continue
        
            return table.get_string()
               
        
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
    
    # First, calculate positional rankings
    position_rankings = {}
    for pos in SKILL_POSITIONS:
        pos_stats = contributions.filter(
            pl.col('position') == pos
        ).groupby(['player_name', 'team']).agg([
            pl.col('overall_contribution').mean().alias('raw_score'),
            pl.col('overall_contribution_adjusted').mean().alias('adjusted_score')
        ]).sort('adjusted_score', descending=True)
        position_rankings[pos] = {
            row['player_name']: rank + 1 
            for rank, row in enumerate(pos_stats.iter_rows(named=True))
        }
    
    markdown += "## Overall Rankings\n\n"
    # Create PrettyTable for top contributors
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Position (Pos. Rank)", "Raw Score", "Adjusted Score", 
                        "Peak Performance", "Trend", "Notable Games"]
    table.align = "l"  # Left align text
    table.float_format = '.1'  # One decimal place for floats
    table.set_style(TableStyle.MARKDOWN)
    
    # Calculate season-long contributions
    contributions = calculate_offensive_shares(team_stats, player_stats, 'overall_contribution')
    
    # Get top 10 by adjusted contribution
    top_contributors = (
        contributions.groupby(['player_name', 'team'])
        .agg([
            pl.col('overall_contribution').mean().alias('raw_score'),
            pl.col('overall_contribution_adjusted').mean().alias('adjusted_score'),
            pl.col('overall_contribution_adjusted').max().alias('peak_score')
        ])
        .sort('adjusted_score', descending=True)
        .head(10)
    )
    
    table = PrettyTable()
    table.field_names = ["Rank", "Player", "Team", "Position (Pos. Rank)", "Raw Score", "Adjusted Score", 
                        "Peak Performance", "Trend", "Notable Games"]
    table.align = "l"  # Left align text
    table.float_format = '.1'  # One decimal place for floats
    
    for rank, row in enumerate(top_contributors.iter_rows(named=True), 1):
        player_name = row['player_name']
        team = row['team']
        position = player_stats.filter(pl.col('player_name') == player_name)['position'].head(1)[0]
        pos_rank = position_rankings[position].get(player_name, "N/A")
        position_display = f"{position} (#{pos_rank})"
        raw_score = row['raw_score']
        adj_score = row['adjusted_score']
        peak = f"Peak: {row['peak_score']:.1f}"
        
        table.add_row([rank, player_name, team, position_display, raw_score, adj_score, peak, "", ""])
    
    return table.get_string()

def process_year(year: int) -> tuple[bool, str, str, str, str]:
    """Process a single year's data with error recovery.
    
    Returns:
        tuple of (success, weekly_markdown, summary_markdown, top_contributors_markdown)
    """
    try:
        logger.info(f"Processing year {year}")
        weekly_markdown = generate_weekly_tables(year)
        summary_markdown = generate_season_summary(year)
        top_contributors = generate_top_contributors(year)
        return True, weekly_markdown, summary_markdown, top_contributors
    except Exception as e:
        logger.error(f"Failed to process year {year}: {str(e)}")
        return False, f"Error processing {year}: {str(e)}", f"Error processing {year}: {str(e)}"

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
                    success, weekly, summary, top_contributors = future.result()
                    results[year] = (success, weekly, summary, top_contributors)
                except Exception as e:
                    logger.error(f"Year {year} failed with error: {str(e)}")
                    results[year] = (False, str(e), str(e), str(e))
    else:
        for year in years:
            success, weekly, summary = process_year(year)
            results[year] = (success, weekly, summary)
    
    # Save results and build index
    successful_years = []
    failed_years = []
    
    for year, (success, weekly, summary, top_contributors) in results.items():
        output_dir = output_base / str(year)
        output_dir.mkdir(exist_ok=True)
        
        if success:
            successful_years.append(year)
            weekly_file = output_dir / "weekly_analysis.md"
            summary_file = output_dir / "season_summary.md"
            top_contributors_file = output_dir / "top_contributors.md"
            weekly_file.write_text(weekly)
            summary_file.write_text(summary)
            top_contributors_file.write_text(top_contributors)
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
            index_markdown += f"- ✅ [{year}](/{year}/)\n"
            index_markdown += f"  - [Weekly Analysis](/{year}/weekly_analysis.md)\n"
            index_markdown += f"  - [Season Summary](/{year}/season_summary.md)\n\n"
    
    if failed_years:
        index_markdown += "## Failed Years\n\n"
        for year in sorted(failed_years):
            index_markdown += f"- ❌ [{year}](/{year}/error.md)\n\n"
    
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
