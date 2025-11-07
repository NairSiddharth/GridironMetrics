"""
Backtest GridironMetrics Props System on Historical Data

Tests the complete props pipeline:
1. Load historical betting lines
2. Generate projections using prop_projection_engine
3. Identify value bets using prop_evaluator
4. Compare against actual outcomes
5. Calculate performance metrics

Usage:
    python backtest_props.py --season 2024 --weeks 1-5
    python backtest_props.py --season 2024 --week 1
"""

import json
import argparse
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from modules.prop_projection_engine import generate_projection
from modules.prop_evaluator import PropEvaluator
from modules.prop_performance_metrics import PropPerformanceMetrics, BetResult
from modules.prop_types import API_MARKET_TO_PROP_TYPE, get_display_name
from modules.logger import get_logger
from modules.constants import CACHE_DIR

logger = get_logger(__name__)


def load_historical_props_for_week(season: int, week: int, day: str = 'friday') -> List[Dict]:
    """
    Load all historical props for a specific week.

    Args:
        season: Season year
        week: Week number
        day: Day of week snapshot (tuesday/friday)

    Returns:
        List of dicts with player prop data
    """
    props_dir = Path(f'cache/player_props/{season}')

    if not props_dir.exists():
        logger.warning(f"No props data directory: {props_dir}")
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

    logger.info(f"Loaded {len(all_props)} player props for {season} Week {week}")
    return all_props


def get_game_info_for_player(player_id: str, season: int, week: int) -> Optional[Tuple[str, str]]:
    """
    Get opponent and team for a player in a specific week.

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Week number

    Returns:
        Tuple of (team, opponent) or None
    """
    # Load schedules to find opponent
    schedule_file = Path(CACHE_DIR) / 'schedules' / f'schedules-{season}.csv'

    if not schedule_file.exists():
        logger.warning(f"Schedule file not found: {schedule_file}")
        return None

    schedule_df = pl.read_csv(schedule_file)

    # Load roster to find team
    roster_file = Path(CACHE_DIR) / 'rosters' / f'rosters-{season}.csv'

    if not roster_file.exists():
        logger.warning(f"Roster file not found: {roster_file}")
        return None

    roster_df = pl.read_csv(roster_file)

    # Find player's team
    player_roster = roster_df.filter(pl.col('gsis_id') == player_id)

    if len(player_roster) == 0:
        return None

    team = player_roster['team'].to_list()[0]

    # Find opponent for this week
    game = schedule_df.filter(
        (pl.col('week') == week) &
        ((pl.col('home_team') == team) | (pl.col('away_team') == team))
    )

    if len(game) == 0:
        return None

    home_team = game['home_team'].to_list()[0]
    away_team = game['away_team'].to_list()[0]
    opponent = away_team if home_team == team else home_team

    return team, opponent


def get_actual_outcome_for_prop(
    player_id: str,
    season: int,
    week: int,
    position: str,
    prop_type: str
) -> Optional[float]:
    """
    Get actual stat outcome from positional player stats cache.

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Week number
        position: Player position
        prop_type: Prop type (e.g., 'passing_yards')

    Returns:
        Actual stat value or None
    """
    # Load positional stats
    stats_file = Path(CACHE_DIR) / 'positional_player_stats' / position.lower() / f'{position.lower()}-{season}.csv'

    if not stats_file.exists():
        logger.warning(f"Stats file not found: {stats_file}")
        return None

    df = pl.read_csv(stats_file)

    # Filter to player and week
    player_week = df.filter(
        (pl.col('player_id') == player_id) &
        (pl.col('week') == week)
    )

    if len(player_week) == 0:
        return None

    # Map prop_type to stat column
    from modules.prop_types import get_stat_column_for_prop
    stat_column = get_stat_column_for_prop(prop_type)

    if not stat_column or stat_column not in player_week.columns:
        return None

    actual = player_week[stat_column].to_list()[0]
    return float(actual) if actual is not None else None


def backtest_week(season: int, week: int, day: str = 'friday', min_edge: float = 0.08) -> List[BetResult]:
    """
    Backtest a single week.

    Args:
        season: Season year
        week: Week number
        day: Day of week snapshot
        min_edge: Minimum edge for value bets (default 8%)

    Returns:
        List of BetResult objects
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Backtesting {season} Week {week} ({day.title()})")
    logger.info(f"{'='*70}")

    # Load historical props
    historical_props = load_historical_props_for_week(season, week, day)

    if not historical_props:
        logger.error(f"No props data found for {season} Week {week}")
        return []

    evaluator = PropEvaluator(min_edge=min_edge)
    bet_results = []

    # Process each player
    for player_data in historical_props:
        player_id = player_data['gsis_id']
        player_name = player_data['player']
        position = player_data['position']

        # Get game info
        game_info = get_game_info_for_player(player_id, season, week)

        if not game_info:
            logger.debug(f"No game info for {player_name} in Week {week}")
            opponent_team = None
        else:
            team, opponent_team = game_info

        # Process each prop
        for prop_display, prop_info in player_data['props'].items():
            betting_line = prop_info['line']
            over_price = prop_info.get('over_price', -110)
            under_price = prop_info.get('under_price', -110)

            # Map API display name to internal prop type
            # This requires reverse lookup from PROP_TYPE_ADJUSTMENTS
            from modules.prop_types import PROP_TYPE_ADJUSTMENTS
            prop_type = None
            for pt, config in PROP_TYPE_ADJUSTMENTS.items():
                if config['display_name'] == prop_display:
                    prop_type = pt
                    break

            if not prop_type:
                logger.warning(f"Unknown prop type: {prop_display}")
                continue

            # Generate projection using prop_projection_engine
            # NOTE: Week for projection should be week-1 (data through previous week)
            try:
                projection_result = generate_projection(
                    player_id=player_id,
                    season=season,
                    week=week - 1,  # Project for week N using data through week N-1
                    position=position,
                    prop_type=prop_type,
                    opponent_team=opponent_team
                )

                if not projection_result:
                    logger.debug(f"No projection for {player_name} {prop_display}")
                    continue

                projection = projection_result['final_projection']

            except Exception as e:
                logger.error(f"Error generating projection for {player_name} {prop_display}: {e}")
                continue

            # Calculate edge
            edge = projection - betting_line
            edge_pct = abs(edge) / betting_line * 100 if betting_line != 0 else 0

            # Only track if edge meets minimum threshold
            if edge_pct < min_edge * 100:
                continue

            # Determine recommendation
            recommendation = 'OVER' if edge > 0 else 'UNDER'

            # Assign confidence grade based on edge
            if edge_pct >= 15:
                confidence = 'A'
            elif edge_pct >= 12:
                confidence = 'B'
            else:
                confidence = 'C'

            # Get actual outcome
            actual_outcome = get_actual_outcome_for_prop(
                player_id=player_id,
                season=season,
                week=week,
                position=position,
                prop_type=prop_type
            )

            # Create BetResult
            bet = BetResult(
                player_id=player_id,
                player_name=player_name,
                position=position,
                prop_type=prop_display,  # Use display name for clarity
                betting_line=betting_line,
                projection=projection,
                actual_outcome=actual_outcome,
                recommendation=recommendation,
                confidence=confidence,
                edge_pct=edge_pct,
                over_price=over_price,
                under_price=under_price
            )

            bet_results.append(bet)

    logger.info(f"Generated {len(bet_results)} bets with edge >= {min_edge*100}%")

    return bet_results


def main():
    """Run backtest on historical data."""

    parser = argparse.ArgumentParser(description='Backtest GridironMetrics props system')
    parser.add_argument('--season', type=int, default=2024, help='Season year')
    parser.add_argument('--week', type=int, help='Single week to test')
    parser.add_argument('--weeks', type=str, help='Week range (e.g., 1-5)')
    parser.add_argument('--day', type=str, default='friday', choices=['tuesday', 'friday'], help='Day snapshot')
    parser.add_argument('--min-edge', type=float, default=8.0, help='Minimum edge % (default 8.0)')

    args = parser.parse_args()

    # Determine weeks to test
    if args.week:
        weeks = [args.week]
    elif args.weeks:
        start, end = map(int, args.weeks.split('-'))
        weeks = list(range(start, end + 1))
    else:
        # Default: test first 5 weeks
        weeks = [1, 2, 3, 4, 5]

    logger.info(f"Backtesting {args.season} Weeks {weeks}")
    logger.info(f"Minimum edge: {args.min_edge}%")
    logger.info(f"Snapshot day: {args.day}")

    all_bets = []

    # Backtest each week
    for week in weeks:
        try:
            week_bets = backtest_week(
                season=args.season,
                week=week,
                day=args.day,
                min_edge=args.min_edge / 100
            )
            all_bets.extend(week_bets)
        except Exception as e:
            logger.error(f"Error backtesting Week {week}: {e}", exc_info=True)
            continue

    if not all_bets:
        logger.error("No bets generated - cannot calculate metrics")
        return

    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTEST RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Total bets analyzed: {len(all_bets)}")

    # Calculate performance metrics
    metrics = PropPerformanceMetrics()

    # Line Hit Rate
    hit_rate = metrics.calculate_line_hit_rate(all_bets)
    logger.info(f"\nLine Hit Rate:")
    logger.info(f"  Overall: {hit_rate['overall_rate']:.1f}% ({hit_rate['overall_wins']}/{hit_rate['overall_total']} bets)")
    logger.info(f"  OVER bets: {hit_rate['over_rate']:.1f}% ({hit_rate['over_wins']}/{hit_rate['over_total']})")
    logger.info(f"  UNDER bets: {hit_rate['under_rate']:.1f}% ({hit_rate['under_wins']}/{hit_rate['under_total']})")

    # By Confidence
    logger.info(f"\n  By Confidence Grade:")
    for grade in ['A', 'B', 'C']:
        if grade in hit_rate['by_confidence']:
            rate = hit_rate['by_confidence'][grade]
            count = hit_rate['by_confidence_counts'][grade]
            logger.info(f"    Grade {grade}: {rate:.1f}% ({count['wins']}/{count['total']} bets)")

    # ROI
    roi = metrics.calculate_roi(all_bets)
    logger.info(f"\nReturn on Investment:")
    logger.info(f"  Overall ROI: {roi['overall_roi']:.1f}%")
    logger.info(f"  Total wagered: ${roi['total_wagered']:.2f}")
    logger.info(f"  Total profit: ${roi['total_profit']:.2f}")
    logger.info(f"  OVER ROI: {roi['over_roi']:.1f}%")
    logger.info(f"  UNDER ROI: {roi['under_roi']:.1f}%")

    # Edge Capture
    edge_capture = metrics.calculate_edge_capture_rate(all_bets)
    logger.info(f"\nEdge Capture Rate:")
    logger.info(f"  8-12% edge: {edge_capture['8-12_rate']:.1f}% ({edge_capture['8-12_count']} bets)")
    logger.info(f"  12-15% edge: {edge_capture['12-15_rate']:.1f}% ({edge_capture['12-15_count']} bets)")
    logger.info(f"  15%+ edge: {edge_capture['15-plus_rate']:.1f}% ({edge_capture['15-plus_count']} bets)")

    # Projection Accuracy
    accuracy = metrics.calculate_projection_accuracy(all_bets)
    logger.info(f"\nProjection Accuracy:")
    logger.info(f"  MAE (Mean Absolute Error): {accuracy['mae']:.2f}")
    logger.info(f"  RMSE (Root Mean Squared Error): {accuracy['rmse']:.2f}")
    logger.info(f"  MAPE (Mean Absolute % Error): {accuracy['mape']:.1f}%")

    # Best bets
    logger.info(f"\nTop 10 Performing Bets:")
    logger.info(f"{'Rank':<6} {'Player':<20} {'Prop':<18} {'Edge':<8} {'Result':<10} {'Profit':<10}")
    logger.info("-" * 80)

    # Sort by profit
    all_bets_sorted = sorted(all_bets, key=lambda b: b.calculate_profit(), reverse=True)

    for i, bet in enumerate(all_bets_sorted[:10], 1):
        profit = bet.calculate_profit()
        result = "WIN" if bet.hit_line() else "LOSS"
        edge_str = f"{bet.edge_pct:.1f}%"
        profit_str = f"${profit:+.2f}"

        logger.info(
            f"{i:<6} {bet.player_name:<20} {bet.prop_type:<18} "
            f"{edge_str:<8} {result:<10} {profit_str:<10}"
        )

    logger.info(f"\n{'='*70}")
    logger.info("Backtest complete!")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
