"""
Historical Player Props Data Fetcher

Fetches historical betting lines from The Odds API for backtesting the
GridironMetrics player props projection system.

Fetches Tuesday (opening lines) and Friday (post-injury lines) for each NFL
week from May 3, 2023 onwards (when player props data became available).

Usage:
    # Fetch single week
    python fetch_historical_props.py --season 2024 --week 1

    # Fetch entire 2024 season
    python fetch_historical_props.py --season 2024

    # Fetch 2024 + partial 2023
    python fetch_historical_props.py --all-seasons

    # Test with single week first
    python fetch_historical_props.py --season 2024 --week 1 --day tuesday

API Usage Estimate (10 credits per event):
    - Single week (Tue/Fri, ~16 games): 1 + (16 × 10) × 2 = ~322 credits
    - 2024 season (18 weeks): 18 × 322 = ~5,796 credits
    - 2023 partial (Weeks 10-18): 9 × 322 = ~2,898 credits
    - Total for both seasons: ~8,694 credits (well within 20K budget)

    Note: Player props available from May 3, 2023 onwards only
"""

import argparse
import logging
from pathlib import Path
from modules.player_props_scraper import PlayerPropsScraper
from modules.logger import get_logger

logger = get_logger(__name__)


def fetch_single_week(
    scraper: PlayerPropsScraper,
    season: int,
    week: int,
    day: str = None,
    filter_to_ranked: bool = False
) -> None:
    """
    Fetch historical props for a single week.

    Args:
        scraper: PlayerPropsScraper instance
        season: Year (e.g., 2024)
        week: Week number (1-18)
        day: 'tuesday', 'friday', or None for both
        filter_to_ranked: Only save players in ranking outputs
    """
    days = [day] if day else ['tuesday', 'friday']

    for d in days:
        try:
            saved_files = scraper.fetch_historical_week_props(
                season=season,
                week=week,
                day=d,
                filter_to_ranked=filter_to_ranked
            )

            logger.info(f"✓ {season} Week {week} {d.title()}: {len(saved_files)} players saved")

        except Exception as e:
            logger.error(f"✗ {season} Week {week} {d.title()} failed: {e}")
            if '--debug' in str(e):
                raise


def fetch_single_season(
    scraper: PlayerPropsScraper,
    season: int,
    start_week: int = 1,
    end_week: int = 18,
    filter_to_ranked: bool = False
) -> None:
    """
    Fetch historical props for an entire season.

    Args:
        scraper: PlayerPropsScraper instance
        season: Year (e.g., 2024)
        start_week: First week to fetch
        end_week: Last week to fetch
        filter_to_ranked: Only save players in ranking outputs
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"FETCHING SEASON {season}")
    logger.info(f"{'='*70}")
    logger.info(f"Weeks {start_week}-{end_week}")

    results = scraper.fetch_historical_season(
        season=season,
        start_week=start_week,
        end_week=end_week,
        days=['tuesday', 'friday'],
        filter_to_ranked=filter_to_ranked
    )

    # Summary
    total_weeks = sum(1 for w in results if any(results[w].values()))
    total_players = sum(
        len(results[w][d])
        for w in results
        for d in results[w]
    )

    logger.info(f"\nSeason {season} Summary:")
    logger.info(f"  Weeks fetched: {total_weeks}/{end_week - start_week + 1}")
    logger.info(f"  Total player files: {total_players}")
    logger.info(f"  API requests remaining: {scraper.requests_remaining}")


def fetch_all_seasons(
    scraper: PlayerPropsScraper,
    seasons: list = None,
    filter_to_ranked: bool = False
) -> None:
    """
    Fetch historical props for multiple seasons.

    Args:
        scraper: PlayerPropsScraper instance
        seasons: List of seasons to fetch (default: 2023-2024 only, as props available from May 2023)
        filter_to_ranked: Only save players in ranking outputs
    """
    if seasons is None:
        seasons = [2023, 2024]  # Only seasons with player props data available

    logger.info(f"\n{'='*70}")
    logger.info(f"FETCHING ALL SEASONS: {seasons}")
    logger.info(f"{'='*70}")
    logger.info(f"Estimated API requests: {len(seasons) * 18 * 2}")
    logger.info(f"Current requests remaining: {scraper.requests_remaining}")

    for season in seasons:
        try:
            fetch_single_season(
                scraper=scraper,
                season=season,
                start_week=1,
                end_week=18,
                filter_to_ranked=filter_to_ranked
            )

            logger.info(f"\n✓ Season {season} complete")

        except Exception as e:
            logger.error(f"\n✗ Season {season} failed: {e}")
            logger.error(f"Stopping at season {season}")
            break

    logger.info(f"\n{'='*70}")
    logger.info("ALL SEASONS FETCH COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Final API requests remaining: {scraper.requests_remaining}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Fetch historical player props from The Odds API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with single week
  python fetch_historical_props.py --season 2024 --week 1 --day tuesday

  # Fetch single week (Tuesday + Friday)
  python fetch_historical_props.py --season 2024 --week 1

  # Fetch entire 2024 season
  python fetch_historical_props.py --season 2024

  # Fetch all available seasons (2023-2024)
  python fetch_historical_props.py --all-seasons

  # Fetch specific weeks from 2023 (props available from Week 10+)
  python fetch_historical_props.py --season 2023 --start-week 10

  # Include all players (not just ranked)
  python fetch_historical_props.py --season 2024 --all-players

API Usage (10 credits per event):
  - Single week: 1 + (~16 games × 10) × 2 days = ~322 credits
  - Single season: 18 weeks × 322 = ~5,796 credits
  - 2023 partial + 2024 full: ~8,694 credits total

Note: Player props data available from May 3, 2023 onwards only
        """
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--season', type=int, help='Fetch single season (or single week if --week specified)')
    mode.add_argument('--all-seasons', action='store_true', help='Fetch all seasons (2020-2024)')

    # Week/day filters
    parser.add_argument('--week', type=int, help='Specific week to fetch (1-18)')
    parser.add_argument('--day', type=str, choices=['tuesday', 'friday'],
                        help='Specific day to fetch (default: both)')
    parser.add_argument('--start-week', type=int, default=1,
                        help='First week to fetch (default: 1)')
    parser.add_argument('--end-week', type=int, default=18,
                        help='Last week to fetch (default: 18)')

    # Season filter (for --all-seasons mode)
    parser.add_argument('--seasons', type=int, nargs='+',
                        help='Specific seasons to fetch (e.g., --seasons 2023 2024)')

    # Player filtering
    parser.add_argument('--all-players', action='store_true',
                        help='Save all players (not just ranked players)')

    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize scraper
    logger.info("Initializing PlayerPropsScraper...")
    scraper = PlayerPropsScraper(rate_limit_delay=1.0)
    logger.info(f"API requests remaining: {scraper.requests_remaining}")

    filter_to_ranked = not args.all_players

    try:
        if args.all_seasons:
            # Fetch all seasons
            seasons = args.seasons if args.seasons else None
            fetch_all_seasons(
                scraper=scraper,
                seasons=seasons,
                filter_to_ranked=filter_to_ranked
            )

        elif args.week:
            # Fetch single week
            fetch_single_week(
                scraper=scraper,
                season=args.season,
                week=args.week,
                day=args.day,
                filter_to_ranked=filter_to_ranked
            )

        else:
            # Fetch single season
            fetch_single_season(
                scraper=scraper,
                season=args.season,
                start_week=args.start_week,
                end_week=args.end_week,
                filter_to_ranked=filter_to_ranked
            )

        logger.info("\n✓ Historical data fetch complete!")

    except KeyboardInterrupt:
        logger.warning("\n\n✗ Fetch interrupted by user")
        logger.info(f"API requests remaining: {scraper.requests_remaining}")

    except Exception as e:
        logger.error(f"\n✗ Fetch failed: {e}")
        if args.debug:
            raise
        logger.info(f"API requests remaining: {scraper.requests_remaining}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
