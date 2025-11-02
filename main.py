"""
main.py

Analyzes NFL offensive skill player shares on a weekly and seasonal basis.
Generates Markdown formatted tables showing offensive contribution percentages.
"""

from pathlib import Path
import polars as pl
from modules.logger import get_logger
from modules.constants import START_YEAR, END_YEAR
from modules.offensive_metrics import OffensiveMetricsCalculator
from modules.play_by_play import PlayByPlayProcessor
from modules.context_adjustments import ContextAdjustments
from modules.personnel_inference import PersonnelInference
from modules.pbp_cache_builder import build_cache
from modules.table_formatters import (
    generate_weekly_tables,
    generate_season_summary,
    generate_qb_rankings,
    generate_rb_rankings,
    generate_wr_rankings,
    generate_te_rankings,
    generate_top_contributors
)
import concurrent.futures
import argparse
import signal
import sys
import atexit

logger = get_logger(__name__)

# Global executor tracking for clean shutdown
_active_executors = []

def cleanup_executors():
    """Clean up all active thread pool executors."""
    global _active_executors
    for executor in _active_executors:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            logger.debug(f"Error shutting down executor: {e}")
    _active_executors.clear()

def signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals."""
    logger.info("\nReceived interrupt signal. Cleaning up processes...")
    cleanup_executors()
    logger.info("Cleanup complete. Exiting.")
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination request

# Windows-specific: handle Ctrl+Break
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal_handler)

# Also register cleanup on normal exit
atexit.register(cleanup_executors)

# Initialize our processors
metrics_calculator = OffensiveMetricsCalculator()
pbp_processor = PlayByPlayProcessor()
context_adj = ContextAdjustments()
personnel_inferencer = PersonnelInference()

# Offensive metrics we want to track for skill positions (RB/WR/TE)

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
        executor = ThreadPoolExecutor(max_workers=min(len(years_to_rebuild), 8))
        _active_executors.append(executor)
        try:
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
        finally:
            executor.shutdown(wait=True)
            if executor in _active_executors:
                _active_executors.remove(executor)
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

    # Clean up old error.md files from previous runs
    for year_dir in output_base.iterdir():
        if year_dir.is_dir():
            error_file = year_dir / "error.md"
            if error_file.exists():
                error_file.unlink()
                logger.debug(f"Removed old error file: {error_file}")
    
    # Process each year
    years = list(range(start_year, end_year + 1))
    
    # Check and rebuild caches upfront if needed
    check_and_rebuild_caches(years, parallel=parallel)
    
    results = {}
    
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=min(len(years), 8))
        _active_executors.append(executor)
        try:
            future_to_year = {executor.submit(process_year, year): year for year in years}
            for future in concurrent.futures.as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    success, weekly, summary, top_overview, top_deep_dive, qb_rankings, rb_rankings, wr_rankings, te_rankings = future.result()
                    results[year] = (success, weekly, summary, top_overview, top_deep_dive, qb_rankings, rb_rankings, wr_rankings, te_rankings)
                except Exception as e:
                    logger.error(f"Year {year} failed with error: {str(e)}")
                    results[year] = (False, str(e), str(e), str(e), str(e), str(e), str(e), str(e), str(e))
        finally:
            executor.shutdown(wait=True)
            if executor in _active_executors:
                _active_executors.remove(executor)
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
