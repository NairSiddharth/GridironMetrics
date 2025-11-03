"""
Rankings Generator Module

Unified ranking generation for all positions (QB, RB, WR, TE).
Consolidates position-specific ranking logic with configurable thresholds and adjustment pipelines.
"""

from pathlib import Path
import polars as pl
from prettytable import PrettyTable, TableStyle
from modules.logger import get_logger
from modules.data_loaders import load_team_weekly_stats, load_position_weekly_stats, is_season_complete, classify_player_profile
from modules.contribution_calculators import (
    calculate_offensive_shares,
    calculate_qb_contribution_from_pbp,
    calculate_average_difficulty,
    adjust_for_opponent
)
from modules.adjustment_pipeline import (
    apply_phase4_adjustments,
    apply_phase4_5_weather_adjustments,
    apply_phase5_adjustments
)

logger = get_logger(__name__)

# Position-specific configuration
POSITION_CONFIG = {
    'QB': {
        'threshold_metric': 'attempts',
        'threshold_value': 14.0,
        'min_game_threshold': 10,  # Min attempts in a game to count it
        'metric_name': 'Pass Att',
        'use_pbp_calculation': True,
        'apply_weather_adjustments': False,
        'table_columns': ["Rank", "QB", "Team", "Games", "Raw", "Def Adj", "Opp Def", "Difficulty", "Avg/Game", "Normalized"]
    },
    'RB': {
        'threshold_metric': 'carries',
        'threshold_value': 6.25,
        'min_game_threshold': None,
        'metric_name': 'Carries',
        'use_pbp_calculation': False,
        'apply_weather_adjustments': True,
        'table_columns': ["Rank", "Player", "Team", "Games", "Raw", "Adj", "Difficulty", "Typical", "Avg/G", "Trend", "Score"]
    },
    'WR': {
        'threshold_metric': 'targets',
        'threshold_value': 3.0,
        'min_game_threshold': None,
        'metric_name': 'Targets',
        'use_pbp_calculation': False,
        'apply_weather_adjustments': True,
        'table_columns': ["Rank", "Player", "Team", "Games", "Raw", "Adj", "Difficulty", "Typical", "Avg/G", "Trend", "Score"]
    },
    'TE': {
        'threshold_metric': 'targets',
        'threshold_value': 2.0,
        'min_game_threshold': None,
        'metric_name': 'Targets',
        'use_pbp_calculation': False,
        'apply_weather_adjustments': True,
        'table_columns': ["Rank", "Player", "Team", "Games", "Raw", "Adj", "Difficulty", "Typical", "Avg/G", "Trend", "Score"]
    }
}


def generate_position_rankings(year: int, position: str) -> str:
    """
    Generate rankings for any position (QB, RB, WR, TE) using unified logic.

    Args:
        year: Season year to generate rankings for
        position: Position code ('QB', 'RB', 'WR', 'TE')

    Returns:
        Markdown-formatted rankings table
    """
    if position not in POSITION_CONFIG:
        return f"Invalid position: {position}. Must be one of: QB, RB, WR, TE"

    config = POSITION_CONFIG[position]
    logger.info(f"Generating {position} rankings for {year}")

    # Load data
    team_stats = load_team_weekly_stats(year)
    if team_stats is None:
        return "No team data available."

    player_stats = load_position_weekly_stats(year, position)
    if player_stats is None:
        return f"No {position} data available."

    # Apply minimum game threshold if configured (QB only)
    if config['min_game_threshold'] is not None:
        player_stats = player_stats.filter(pl.col(config['threshold_metric']) >= config['min_game_threshold'])

    # Calculate games and threshold metric per game
    player_games = player_stats.group_by(['player_id', 'player_name']).agg([
        pl.col('week').count().alias('games'),
        pl.col(config['threshold_metric']).sum().alias(f"total_{config['threshold_metric']}")
    ])

    player_games = player_games.with_columns([
        (pl.col(f"total_{config['threshold_metric']}") / pl.col('games')).alias(f"{config['threshold_metric']}_per_game")
    ])

    # Filter to qualified players
    qualified_players = player_games.filter(pl.col(f"{config['threshold_metric']}_per_game") >= config['threshold_value'])
    player_stats = player_stats.join(
        qualified_players.select(['player_id', 'player_name']),
        on=['player_id', 'player_name'],
        how='inner'
    )

    if len(player_stats) == 0:
        return f"No qualified {position}s ({config['threshold_value']}+ {config['metric_name'].lower()}/game) for this season."

    logger.info(f"Filtered to {len(qualified_players)} qualified {position}s ({config['threshold_value']}+ {config['metric_name'].lower()}/game)")

    # Branch based on calculation method
    if config['use_pbp_calculation']:
        # QB-specific calculation using play-by-play data
        return _generate_qb_rankings_internal(year, player_stats, team_stats, qualified_players, config)
    else:
        # Skill position calculation using offensive shares
        return _generate_skill_position_rankings_internal(year, position, player_stats, team_stats, qualified_players, config)


def _generate_qb_rankings_internal(year: int, qb_stats: pl.DataFrame, team_stats: pl.DataFrame,
                                   qualified_qbs: pl.DataFrame, config: dict) -> str:
    """Internal function for QB-specific ranking generation."""

    # Calculate QB contribution scores using play-by-play data
    qb_contributions = []

    # Check if FTN data is available
    from modules.ftn_cache_builder import FTN_START_YEAR
    has_ftn = year >= FTN_START_YEAR

    if has_ftn:
        logger.info(f"Calculating QB contributions with FTN contextual adjustments (PA, OOP, blitz, screen)...")
    else:
        logger.info(f"Calculating QB contributions from play-by-play data with contextual penalties and pressure bonuses...")

    for qb_name in qb_stats['player_name'].unique().to_list():
        qb_data = qb_stats.filter(pl.col('player_name') == qb_name)
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

    # Calculate average opponent defense multipliers
    qb_weekly_with_defense = adjust_for_opponent(qb_stats, team_stats)
    avg_opponent_defense = qb_weekly_with_defense.group_by(['player_id', 'player_name']).agg([
        pl.col('pass_defense_multiplier').mean().alias('avg_opponent_defense_mult')
    ])

    qb_season = qb_season.join(
        avg_opponent_defense.select(['player_id', 'avg_opponent_defense_mult']),
        on='player_id',
        how='left'
    ).with_columns([
        pl.col('avg_opponent_defense_mult').fill_null(1.0).clip(0.5, 2.5).alias('avg_opponent_defense_mult')
    ])

    # Apply opponent defense adjustment
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

    # Calculate team-level max games
    team_games = qb_stats.group_by('team').agg([
        pl.col('week').n_unique().alias('team_max_games')
    ])

    qb_season = qb_season.join(team_games, on='team', how='left')

    # Calculate hybrid score
    qb_season = qb_season.with_columns([
        (pl.col('defense_adjusted_score') * (pl.col('games') / pl.col('team_max_games')).pow(0.4)).alias('hybrid_score')
    ])

    # Z-score normalization
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
    table.field_names = config['table_columns']
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)

    for rank, row in enumerate(qb_season.iter_rows(named=True), 1):
        table.add_row([
            rank,
            row['player_name'],
            row['team'].upper(),
            row['games'],
            f"{row['raw_score']:.2f}",
            f"{row['defense_adjusted_score']:.2f}",
            f"{row.get('avg_opponent_defense_mult', 1.0):.3f}",
            f"{row.get('avg_difficulty_multiplier', 1.0):.3f}",
            f"{row['avg_per_game']:.2f}",
            f"{row['normalized_hybrid']:.2f}"
        ])

    markdown += table.get_string() + "\n\n"
    return markdown


def _generate_skill_position_rankings_internal(year: int, position: str, player_stats: pl.DataFrame,
                                               team_stats: pl.DataFrame, qualified_players: pl.DataFrame,
                                               config: dict) -> str:
    """Internal function for skill position (RB/WR/TE) ranking generation."""

    # Calculate contributions with and without situational adjustments
    raw_contributions = calculate_offensive_shares(
        team_stats,
        player_stats,
        'overall_contribution',
        opponent_stats=team_stats,
        year=None  # No situational adjustments for raw score
    )

    adjusted_contributions = calculate_offensive_shares(
        team_stats,
        player_stats,
        'overall_contribution',
        opponent_stats=team_stats,
        year=year  # Enable situational adjustments
    )

    # Add position column for Phase 4 adjustments
    adjusted_contributions = adjusted_contributions.with_columns([
        pl.lit(position).alias('position')
    ])

    # Apply Phase 4 adjustments (catch rate + blocking quality)
    adjusted_contributions = apply_phase4_adjustments(adjusted_contributions, year)

    # Apply Phase 4.5 weather adjustments if configured
    if config['apply_weather_adjustments']:
        adjusted_contributions = apply_phase4_5_weather_adjustments(adjusted_contributions, year, position)

    # Save per-game contributions before Phase 5
    per_game_contributions = adjusted_contributions.clone().select([
        'player_id', 'player_name', 'team', 'week', 'player_overall_contribution'
    ])

    # Apply Phase 5 adjustments (talent context + sample size dampening)
    adjusted_contributions = apply_phase5_adjustments(adjusted_contributions, year)

    # Aggregate to season totals
    raw_season = raw_contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').sum().alias('raw_score'),
        pl.col('week').count().alias('games')
    ])

    adjusted_season = adjusted_contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').sum().alias('adjusted_score')
    ])

    # Join raw and adjusted
    season_totals = raw_season.join(adjusted_season, on=['player_id', 'player_name', 'team'], how='left')

    # Calculate raw average per game (for Avg/G column)
    season_totals = season_totals.with_columns([
        (pl.col('raw_score') / pl.col('games')).alias('raw_avg_per_game')
    ])

    # Calculate average difficulty multiplier
    avg_difficulty = calculate_average_difficulty(year, player_stats)
    if avg_difficulty is not None:
        season_totals = season_totals.join(
            avg_difficulty.select(['player_id', 'player_name', 'avg_difficulty_multiplier']),
            on=['player_id', 'player_name'],
            how='left'
        )
    else:
        season_totals = season_totals.with_columns([
            pl.lit(1.0).alias('avg_difficulty_multiplier')
        ])

    # Calculate consistency metrics
    season_totals = _calculate_consistency_metrics(
        season_totals, per_game_contributions, adjusted_contributions, player_stats, year
    )

    # Calculate final score: adjusted * difficulty * sqrt(games/16)
    season_totals = season_totals.with_columns([
        (pl.col('adjusted_score') * pl.col('avg_difficulty_multiplier') * (pl.col('games') / 16).sqrt()).alias('final_score')
    ])

    # Sort by final score
    season_totals = season_totals.sort('final_score', descending=True)

    # Create table
    markdown = f"# {position} Rankings - {year}\n\n"
    table = PrettyTable()
    table.field_names = config['table_columns']
    table.align = "l"
    table.float_format = '.2'
    table.set_style(TableStyle.MARKDOWN)

    for rank, row in enumerate(season_totals.iter_rows(named=True), 1):
        table.add_row([
            rank,
            row['player_name'],
            row['team'].upper(),
            row['games'],
            f"{row['raw_score']:.2f}",
            f"{row['adjusted_score']:.2f}",
            f"{row.get('avg_difficulty_multiplier', 1.0):.3f}",
            f"{row.get('typical', 0):.2f}",
            f"{row['raw_avg_per_game']:.2f}",
            row.get('trend', 'N/A'),
            f"{row['final_score']:.2f}"
        ])

    markdown += table.get_string() + "\n\n"
    return markdown


def _calculate_consistency_metrics(season_totals: pl.DataFrame, per_game_contributions: pl.DataFrame,
                                   adjusted_contributions: pl.DataFrame, player_stats: pl.DataFrame,
                                   year: int) -> pl.DataFrame:
    """Calculate consistency metrics (boom/bust/steady games) for players."""

    # Get Phase 5 scaling factors
    pre_phase5_means = per_game_contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').mean().alias('pre_phase5_mean')
    ])

    post_phase5_values = adjusted_contributions.group_by(['player_id', 'player_name', 'team']).agg([
        pl.col('player_overall_contribution').mean().alias('post_phase5_value')
    ])

    scaling_factors = pre_phase5_means.join(post_phase5_values, on=['player_id', 'player_name', 'team'])
    scaling_factors = scaling_factors.with_columns([
        (pl.col('post_phase5_value') / pl.col('pre_phase5_mean')).alias('phase5_scaling_factor')
    ])

    # Calculate consistency metrics for each player
    consistency_data = []
    for player_id in season_totals['player_id'].to_list():
        player_weeks = per_game_contributions.filter(pl.col('player_id') == player_id)
        if len(player_weeks) == 0:
            continue

        contributions = player_weeks['player_overall_contribution'].to_list()
        contributions.sort()

        # Get Phase 5 scaling factor
        player_scaling = scaling_factors.filter(pl.col('player_id') == player_id)
        scale_factor = player_scaling['phase5_scaling_factor'][0] if len(player_scaling) > 0 else 1.0

        # Calculate typical (25th/75th percentile average) with Phase 5 scaling
        if len(contributions) >= 4:
            q1_idx = len(contributions) // 4
            q3_idx = 3 * len(contributions) // 4
            typical = (contributions[q1_idx] + contributions[q3_idx]) / 2 * scale_factor
        else:
            typical = sum(contributions) / len(contributions) * scale_factor if contributions else 0

        # Calculate boom/bust/steady counts
        mean_contrib = sum(contributions) / len(contributions)
        variance = sum((x - mean_contrib) ** 2 for x in contributions) / len(contributions)
        std_dev = variance ** 0.5

        bust_threshold = mean_contrib - (0.5 * std_dev)
        boom_threshold = mean_contrib + (0.5 * std_dev)

        below_avg = sum(1 for g in contributions if g < bust_threshold)
        at_avg = sum(1 for g in contributions if bust_threshold <= g <= boom_threshold)
        above_avg = sum(1 for g in contributions if g > boom_threshold)

        # Determine trend based on season completion
        max_week_in_season = player_stats['week'].max()
        if is_season_complete(year):
            trend_str = classify_player_profile(above_avg, at_avg, below_avg)
        else:
            trend_str = f"Week {max_week_in_season}"

        consistency_data.append({
            'player_id': player_id,
            'typical': typical,
            'trend': trend_str
        })

    if consistency_data:
        consistency_df = pl.DataFrame(consistency_data)
        season_totals = season_totals.join(consistency_df, on='player_id', how='left')
    else:
        season_totals = season_totals.with_columns([
            pl.lit(0.0).alias('typical'),
            pl.lit('N/A').alias('trend')
        ])

    return season_totals
