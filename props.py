"""
props.py

GridironMetrics Player Props System

CLI interface for generating prop projections and identifying value bets.

Usage:
    python props.py project 2024 11 --player-id 00-0033873
    python props.py evaluate 2024 11 --lines-file cache/betting_lines/week11.json
    python props.py validate 2024 10  # Validate Week 10 projections against actuals

Commands:
    project - Generate projection for a single player
    evaluate - Evaluate all props for a week against betting lines
    validate - Validate projections against actual outcomes
"""

import argparse
import json
import sys
from pathlib import Path
from modules.logger import get_logger
from modules.prop_projection_engine import generate_projection
from modules.prop_evaluator import PropEvaluator
from modules.prop_output_formatter import PropOutputFormatter
from modules.prop_types import get_prop_types_for_position
import polars as pl

logger = get_logger(__name__)


def command_project(args):
    """Generate projection for a single player."""
    logger.info(f"Generating projection for player {args.player_id}")

    # Get player info from roster
    roster_file = Path(f"cache/rosters/rosters-{args.season}.csv")
    if not roster_file.exists():
        logger.error(f"Roster file not found: {roster_file}")
        return 1

    roster = pl.read_csv(roster_file)
    player = roster.filter(pl.col('gsis_id') == args.player_id)

    if len(player) == 0:
        logger.error(f"Player not found: {args.player_id}")
        return 1

    player_name = player['full_name'][0]
    position = player['position'][0]

    logger.info(f"Player: {player_name} ({position})")

    # Get prop types for position
    prop_types = get_prop_types_for_position(position)

    if args.prop_type:
        if args.prop_type not in prop_types:
            logger.error(f"Invalid prop type '{args.prop_type}' for {position}")
            logger.info(f"Valid prop types: {', '.join(prop_types)}")
            return 1
        prop_types = [args.prop_type]

    # Generate projections
    print(f"\n{'='*70}")
    print(f"{player_name} ({position}) - Week {args.week} Projections")
    print('='*70)

    for prop_type in prop_types:
        projection = generate_projection(
            player_id=args.player_id,
            season=args.season,
            week=args.week,
            position=position,
            prop_type=prop_type,
            opponent_team=args.opponent,
            game_weather=None  # TODO: Parse weather args if needed
        )

        if not projection:
            print(f"\n{prop_type}: No projection available")
            continue

        print(f"\n{projection['prop_type'].upper().replace('_', ' ')}:")
        print(f"  Baseline: {projection['baseline']:.1f}")
        print(f"  Adjusted: {projection['adjusted_projection']:.1f}")
        print(f"  Final (dampened): {projection['final_projection']:.1f}")
        print(f"  Effective games: {projection['effective_games']:.1f}")
        print(f"\n  Adjustments:")
        for adj_name, multiplier in projection['adjustments'].items():
            print(f"    {adj_name}: {multiplier:.3f}x")

        if projection['stat_summary']:
            summary = projection['stat_summary']
            print(f"\n  Historical:")
            print(f"    Last 3 avg: {summary['last_3_avg']:.1f}")
            print(f"    Last 5 avg: {summary['last_5_avg']:.1f}")
            print(f"    Season avg: {summary['season_avg']:.1f}")
            print(f"    Variance (CV): {summary['variance']:.3f}")

    print()
    return 0


def command_evaluate(args):
    """Evaluate props for a week against betting lines."""
    logger.info(f"Evaluating Week {args.week} props")

    # Load betting lines
    lines_file = Path(args.lines_file)
    if not lines_file.exists():
        logger.error(f"Betting lines file not found: {lines_file}")
        logger.info("Expected format: JSON file with betting lines data")
        return 1

    with open(lines_file, 'r') as f:
        betting_lines_data = json.load(f)

    logger.info(f"Loaded betting lines for {len(betting_lines_data)} players")

    # Initialize evaluator
    evaluator = PropEvaluator(min_edge=args.min_edge)

    # Evaluate all props
    value_bets, summary = evaluator.evaluate_week(
        season=args.season,
        week=args.week,
        betting_lines_data=betting_lines_data
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"Week {args.week} Evaluation Summary")
    print('='*70)
    print(f"Total props evaluated: {summary['total_props_evaluated']}")
    print(f"Value bets found: {summary['total_value_found']} ({summary['value_pct']:.1f}%)")
    print(f"Average edge: {summary['avg_edge_pct']:.1f}%")
    print(f"\nConfidence breakdown:")
    print(f"  Grade A: {summary['confidence_breakdown']['A']}")
    print(f"  Grade B: {summary['confidence_breakdown']['B']}")
    print(f"  Grade C: {summary['confidence_breakdown']['C']}")
    print(f"\nRecommendations:")
    print(f"  OVER: {summary['recommendation_breakdown']['OVER']}")
    print(f"  UNDER: {summary['recommendation_breakdown']['UNDER']}")

    # Print top value bets
    if value_bets:
        print(f"\n{'='*70}")
        print("Top 10 Value Bets")
        print('='*70)
        for i, bet in enumerate(value_bets[:10], 1):
            edge_str = f"{bet['edge_pct']*100:+.1f}%"
            print(
                f"{i}. {bet['player_name']} ({bet['position']}) - "
                f"{bet['prop_display']}: {bet['betting_line']:.1f} → {bet['projection']:.1f} "
                f"({edge_str} edge, {bet['recommendation']}, Grade {bet['confidence']})"
            )

    # Save output if requested
    if not args.no_output:
        formatter = PropOutputFormatter(Path("output"))

        # Group props by position for output
        all_props_by_position = {'QB': [], 'RB': [], 'WR': [], 'TE': []}
        for bet in value_bets:
            if bet['position'] in all_props_by_position:
                all_props_by_position[bet['position']].append(bet)

        formatter.save_week_output(
            season=args.season,
            week=args.week,
            value_bets=value_bets,
            summary=summary,
            all_props_by_position=all_props_by_position
        )

        output_dir = Path("output") / str(args.season) / "playerprops" / f"week{args.week}"
        logger.info(f"Output saved to {output_dir}")

    print()
    return 0


def command_validate(args):
    """Validate projections against actual outcomes."""
    logger.info(f"Validating Week {args.week} projections against actuals")

    # Run the validation test
    from tests.validate_week10 import validate_player_projections
    import polars as pl

    # Load QB stats
    qb_stats = pl.read_csv(f"cache/positional_player_stats/qb/qb-{args.season}.csv")

    # Get QBs who played in this week
    week_qbs = qb_stats.filter(pl.col('week') == args.week).sort('passing_yards', descending=True).head(args.num_players)

    print(f"\n{'='*70}")
    print(f"Week {args.week} Projection Validation")
    print('='*70)
    print(f"Testing top {len(week_qbs)} QBs by passing yards\n")

    # Validate each QB
    for i in range(len(week_qbs)):
        player_name = week_qbs['player_display_name'][i]
        player_id = week_qbs['player_id'][i]

        validate_player_projections(
            player_id=player_id,
            player_name=player_name,
            position='QB',
            season=args.season,
            week=args.week
        )

    print()
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GridironMetrics Player Props System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate projection for Patrick Mahomes
  python props.py project 2024 11 --player-id 00-0033873 --opponent BUF

  # Evaluate all props for Week 11
  python props.py evaluate 2024 11 --lines-file cache/betting_lines/week11.json

  # Validate Week 10 projections
  python props.py validate 2024 10
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Project command
    project_parser = subparsers.add_parser('project', help='Generate projection for a player')
    project_parser.add_argument('season', type=int, help='Season year')
    project_parser.add_argument('week', type=int, help='Week number')
    project_parser.add_argument('--player-id', required=True, help='Player GSIS ID')
    project_parser.add_argument('--prop-type', help='Specific prop type (optional, default: all)')
    project_parser.add_argument('--opponent', help='Opponent team abbreviation')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate props vs betting lines')
    evaluate_parser.add_argument('season', type=int, help='Season year')
    evaluate_parser.add_argument('week', type=int, help='Week number')
    evaluate_parser.add_argument('--lines-file', required=True, help='Betting lines JSON file')
    evaluate_parser.add_argument('--min-edge', type=float, default=0.08, help='Minimum edge threshold (default: 0.08 = 8%%)')
    evaluate_parser.add_argument('--no-output', action='store_true', help='Skip saving output files')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate projections vs actuals')
    validate_parser.add_argument('season', type=int, help='Season year')
    validate_parser.add_argument('week', type=int, help='Week number')
    validate_parser.add_argument('--num-players', type=int, default=5, help='Number of players to validate (default: 5)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command
    if args.command == 'project':
        return command_project(args)
    elif args.command == 'evaluate':
        return command_evaluate(args)
    elif args.command == 'validate':
        return command_validate(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
