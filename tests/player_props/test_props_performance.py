"""
Test Player Props Performance Against Historical Data

Analyzes how well GridironMetrics projections performed against:
1. Historical betting lines (did we beat the line?)
2. Actual player performance (how accurate were projections?)

Uses proper betting metrics: Line Hit Rate, ROI, Edge Capture Rate
"""

import json
import nflreadpy as nfl
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from modules.prop_performance_metrics import PropPerformanceMetrics, BetResult
from modules.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PropComparison:
    """Single comparison: projection vs line vs actual"""
    player_id: str
    player_name: str
    position: str
    season: int
    week: int
    day: str  # tuesday or friday
    prop_type: str
    betting_line: float
    projection: Optional[float]
    actual_outcome: Optional[float]
    over_price: int
    under_price: int
    game_id: Optional[str] = None


def load_historical_props(season: int, week: int, day: str) -> List[Dict]:
    """Load all historical props for a specific week/day"""
    props_dir = Path(f'cache/player_props/{season}')

    if not props_dir.exists():
        logger.warning(f"No props data for {season}")
        return []

    all_props = []

    # Iterate through all player directories
    for player_dir in props_dir.iterdir():
        if not player_dir.is_dir():
            continue

        week_dir = player_dir / f'week{week}'
        if not week_dir.exists():
            continue

        day_file = week_dir / f'{day}.json'
        if not day_file.exists():
            continue

        try:
            with open(day_file, 'r') as f:
                props_data = json.load(f)
                all_props.append(props_data)
        except Exception as e:
            logger.error(f"Error loading {day_file}: {e}")
            continue

    return all_props


def load_player_rankings(season: int, week: int) -> pd.DataFrame:
    """Load player rankings/projections for a specific week"""
    rankings_file = Path(f'output/{season}/week{week}/player_rankings.csv')

    if not rankings_file.exists():
        logger.warning(f"No rankings found for {season} Week {week}")
        return pd.DataFrame()

    df = pd.read_csv(rankings_file)
    return df


def get_actual_stats(season: int, week: int) -> pd.DataFrame:
    """Get actual player stats from nfl_data_py for a specific week"""
    logger.info(f"Loading actual stats for {season} Week {week}...")

    # Load weekly stats
    weekly_df = nfl.import_weekly_data([season])

    # Filter to specific week
    weekly_df = weekly_df[weekly_df['week'] == week].copy()

    return weekly_df


def match_projection_to_prop(player_id: str, position: str, prop_type: str,
                             rankings_df: pd.DataFrame) -> Optional[float]:
    """
    Extract projection value for a specific prop type from rankings

    Prop Type Mapping:
    - Passing Yards -> passing_yards
    - Passing TDs -> passing_tds
    - Rushing Yards -> rushing_yards
    - Receptions -> receptions
    - Receiving Yards -> receiving_yards
    """
    if rankings_df.empty:
        return None

    # Find player in rankings
    player_row = rankings_df[rankings_df['gsis_id'] == player_id]

    if player_row.empty:
        return None

    # Map prop type to column name
    prop_column_map = {
        'Passing Yards': 'passing_yards',
        'Passing TDs': 'passing_tds',
        'Rushing Yards': 'rushing_yards',
        'Receptions': 'receptions',
        'Receiving Yards': 'receiving_yards',
    }

    column = prop_column_map.get(prop_type)

    if column and column in player_row.columns:
        value = player_row[column].iloc[0]
        return float(value) if pd.notna(value) else None

    return None


def get_actual_outcome(player_id: str, prop_type: str, stats_df: pd.DataFrame) -> Optional[float]:
    """Extract actual stat value for a specific prop type"""
    if stats_df.empty:
        return None

    # Find player in stats
    player_stats = stats_df[stats_df['player_id'] == player_id]

    if player_stats.empty:
        return None

    # Map prop type to stat column
    stat_column_map = {
        'Passing Yards': 'passing_yards',
        'Passing TDs': 'passing_tds',
        'Rushing Yards': 'rushing_yards',
        'Receptions': 'receptions',
        'Receiving Yards': 'receiving_yards',
    }

    column = stat_column_map.get(prop_type)

    if column and column in player_stats.columns:
        value = player_stats[column].iloc[0]
        return float(value) if pd.notna(value) else None

    return None


def analyze_week(season: int, week: int, day: str = 'friday') -> List[PropComparison]:
    """Analyze all props for a specific week"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing {season} Week {week} ({day.title()})")
    logger.info(f"{'='*70}")

    # Load data
    props = load_historical_props(season, week, day)
    rankings = load_player_rankings(season, week)
    stats = get_actual_stats(season, week)

    logger.info(f"Loaded {len(props)} player props")
    logger.info(f"Loaded rankings: {len(rankings)} players")
    logger.info(f"Loaded stats: {len(stats)} player-games")

    comparisons = []

    for prop_data in props:
        player_id = prop_data['gsis_id']
        player_name = prop_data['player']
        position = prop_data['position']

        for prop_type, prop_info in prop_data['props'].items():
            line = prop_info['line']
            over_price = prop_info.get('over_price', -110)
            under_price = prop_info.get('under_price', -110)

            # Get projection
            projection = match_projection_to_prop(player_id, position, prop_type, rankings)

            # Get actual outcome
            actual = get_actual_outcome(player_id, prop_type, stats)

            comparison = PropComparison(
                player_id=player_id,
                player_name=player_name,
                position=position,
                season=season,
                week=week,
                day=day,
                prop_type=prop_type,
                betting_line=line,
                projection=projection,
                actual_outcome=actual,
                over_price=over_price,
                under_price=under_price
            )

            comparisons.append(comparison)

    logger.info(f"Created {len(comparisons)} prop comparisons")

    return comparisons


def generate_bet_recommendations(comparisons: List[PropComparison],
                                min_edge: float = 8.0) -> List[BetResult]:
    """
    Generate bet recommendations based on projection vs line

    Args:
        comparisons: List of prop comparisons
        min_edge: Minimum edge percentage to recommend a bet (default 8%)

    Returns:
        List of BetResult objects for recommended bets
    """
    bets = []

    for comp in comparisons:
        if comp.projection is None or comp.actual_outcome is None:
            continue

        # Calculate edge
        edge_pct = abs(comp.projection - comp.betting_line) / comp.betting_line * 100

        if edge_pct < min_edge:
            continue  # Not enough edge

        # Determine recommendation
        if comp.projection > comp.betting_line:
            recommendation = 'OVER'
        else:
            recommendation = 'UNDER'

        # Assign confidence grade based on edge
        if edge_pct >= 15:
            confidence = 'A'
        elif edge_pct >= 12:
            confidence = 'B'
        else:
            confidence = 'C'

        bet = BetResult(
            player_id=comp.player_id,
            player_name=comp.player_name,
            position=comp.position,
            prop_type=comp.prop_type,
            betting_line=comp.betting_line,
            projection=comp.projection,
            actual_outcome=comp.actual_outcome,
            recommendation=recommendation,
            confidence=confidence,
            edge_pct=edge_pct,
            over_price=comp.over_price,
            under_price=comp.under_price
        )

        bets.append(bet)

    return bets


def main():
    """Run performance analysis on historical data"""

    # Test configuration
    test_weeks = [
        (2024, 1, 'friday'),
        (2024, 2, 'friday'),
        (2024, 3, 'friday'),
        (2024, 4, 'friday'),
        (2024, 5, 'friday'),
        (2023, 1, 'friday'),
        (2023, 2, 'friday'),
    ]

    all_bets = []

    for season, week, day in test_weeks:
        try:
            comparisons = analyze_week(season, week, day)
            bets = generate_bet_recommendations(comparisons, min_edge=8.0)
            all_bets.extend(bets)

            logger.info(f"Generated {len(bets)} bet recommendations for {season} Week {week}")

        except Exception as e:
            logger.error(f"Error analyzing {season} Week {week}: {e}")
            continue

    if not all_bets:
        logger.error("No bets generated - cannot calculate metrics")
        return

    logger.info(f"\n{'='*70}")
    logger.info(f"OVERALL PERFORMANCE METRICS")
    logger.info(f"{'='*70}")
    logger.info(f"Total bets analyzed: {len(all_bets)}")

    # Calculate metrics
    metrics = PropPerformanceMetrics()

    # Line Hit Rate
    hit_rate = metrics.calculate_line_hit_rate(all_bets)
    logger.info(f"\nLine Hit Rate:")
    logger.info(f"  Overall: {hit_rate['overall_rate']:.1f}% ({hit_rate['overall_wins']}/{hit_rate['overall_total']} bets)")
    logger.info(f"  OVER bets: {hit_rate['over_rate']:.1f}%")
    logger.info(f"  UNDER bets: {hit_rate['under_rate']:.1f}%")

    # By Confidence
    for grade in ['A', 'B', 'C']:
        if grade in hit_rate['by_confidence']:
            rate = hit_rate['by_confidence'][grade]
            logger.info(f"  Grade {grade}: {rate:.1f}%")

    # ROI
    roi = metrics.calculate_roi(all_bets)
    logger.info(f"\nReturn on Investment:")
    logger.info(f"  Overall ROI: {roi['overall_roi']:.1f}%")
    logger.info(f"  Total wagered: ${roi['total_wagered']:.2f}")
    logger.info(f"  Total profit: ${roi['total_profit']:.2f}")

    # Edge Capture
    edge_capture = metrics.calculate_edge_capture_rate(all_bets)
    logger.info(f"\nEdge Capture Rate:")
    for bucket, rate in edge_capture.items():
        if 'rate' in bucket:
            logger.info(f"  {bucket}: {rate:.1f}%")

    # Projection Accuracy
    accuracy = metrics.calculate_projection_accuracy(all_bets)
    logger.info(f"\nProjection Accuracy:")
    logger.info(f"  MAE (Mean Absolute Error): {accuracy['mae']:.2f}")
    logger.info(f"  RMSE (Root Mean Squared Error): {accuracy['rmse']:.2f}")

    logger.info(f"\n{'='*70}")
    logger.info("Analysis complete!")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
