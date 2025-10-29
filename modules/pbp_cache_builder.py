"""
pbp_cache_builder.py

Builds a cache of play-by-play data with situational multipliers pre-calculated.
Run this once to generate cache files, then main.py can load from cache instead of 
processing PBP data on every run.

Usage:
    python -m modules.pbp_cache_builder --start-year 2000 --end-year 2025
    python -m modules.pbp_cache_builder --year 2021 --force
"""

import polars as pl
from pathlib import Path
from typing import Optional
import nflreadpy as nfl
import argparse
from .logger import get_logger
from .constants import CACHE_DIR, START_YEAR, END_YEAR

logger = get_logger(__name__)


def get_cache_path(year: int, cache_root: Path = None) -> Path:
    """Get the cache file path for a given year."""
    if cache_root is None:
        cache_root = Path(CACHE_DIR) / "pbp"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"pbp_{year}.parquet"


def load_and_process_pbp(year: int) -> Optional[pl.DataFrame]:
    """
    Load PBP data from nflreadpy and calculate all situational multipliers.
    
    Args:
        year: Season year to process
        
    Returns:
        DataFrame with PBP data and pre-calculated multipliers, or None on error
    """
    try:
        logger.info(f"Loading play-by-play data for {year} from nflreadpy...")
        pbp_data = nfl.load_pbp(seasons=year)
        
        # Filter to regular season only
        if pbp_data is not None and "season_type" in pbp_data.columns:
            pbp_data = pbp_data.filter(pl.col("season_type") == "REG")
        else:
            logger.error(f"No PBP data or season_type column missing for {year}")
            return None
            
        logger.info(f"Loaded {len(pbp_data)} plays for {year}")
        
        # Calculate all situational multipliers once
        logger.info(f"Calculating situational multipliers for {year}...")
        pbp_data = pbp_data.with_columns([
            # Field position multipliers
            pl.when(pl.col("yardline_100") <= 5)
            .then(2.0)  # Goalline (highest value)
            .when(pl.col("yardline_100") <= 20)
            .then(1.5)  # Redzone
            .otherwise(1.0)
            .alias("field_position_multiplier"),
            
            # Score differential multipliers
            pl.when(pl.col("score_differential").abs() <= 8)
            .then(1.5)  # One score game
            .when(pl.col("score_differential").abs() <= 16)
            .then(1.25)  # Two score game
            .otherwise(1.0)
            .alias("score_multiplier"),
            
            # Time remaining multipliers (last 2 minutes of each quarter)
            pl.when((pl.col("qtr") == 4) & (pl.col("quarter_seconds_remaining") <= 120))
            .then(1.5)  # 4th quarter critical time (highest)
            .when((pl.col("qtr") == 3) & (pl.col("quarter_seconds_remaining") <= 120))
            .then(1.3)  # 3rd quarter critical time
            .when((pl.col("qtr") == 2) & (pl.col("quarter_seconds_remaining") <= 120))
            .then(1.2)  # 2nd quarter critical time (2-minute drill)
            .when((pl.col("qtr") == 1) & (pl.col("quarter_seconds_remaining") <= 120))
            .then(1.1)  # 1st quarter critical time (lowest)
            .otherwise(1.0)
            .alias("time_multiplier"),
            
            # Down multipliers
            pl.when(pl.col("down") == 4)
            .then(1.5)  # 4th down
            .when(pl.col("down") == 3)
            .then(1.25)  # 3rd down
            .otherwise(1.0)
            .alias("down_multiplier")
        ]).with_columns([
            # Combined situational multiplier
            (pl.col("field_position_multiplier") * 
             pl.col("score_multiplier") * 
             pl.col("time_multiplier") * 
             pl.col("down_multiplier")).alias("situational_multiplier")
        ])
        
        logger.info(f"Calculated multipliers for {len(pbp_data)} plays")
        return pbp_data
        
    except Exception as e:
        logger.error(f"Error loading/processing PBP data for {year}: {str(e)}")
        return None


def build_cache(year: int, force: bool = False) -> bool:
    """
    Build cache file for a specific year.
    
    Args:
        year: Season year to cache
        force: If True, rebuild even if cache exists
        
    Returns:
        True if successful, False otherwise
    """
    cache_path = get_cache_path(year)
    
    # Check if cache already exists
    if cache_path.exists() and not force:
        logger.info(f"Cache already exists for {year}: {cache_path}")
        return True
    
    # Load and process PBP data
    pbp_data = load_and_process_pbp(year)
    if pbp_data is None:
        return False
    
    # Save to parquet
    try:
        logger.info(f"Saving cache for {year} to {cache_path}...")
        pbp_data.write_parquet(cache_path)
        logger.info(f"Successfully cached {year} PBP data ({len(pbp_data)} plays)")
        return True
    except Exception as e:
        logger.error(f"Error saving cache for {year}: {str(e)}")
        return False


def build_cache_range(start_year: int, end_year: int, force: bool = False) -> dict:
    """
    Build cache files for a range of years.
    
    Args:
        start_year: First year to cache
        end_year: Last year to cache (inclusive)
        force: If True, rebuild even if cache exists
        
    Returns:
        Dictionary with year: success status
    """
    results = {}
    for year in range(start_year, end_year + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing year {year} ({year - start_year + 1}/{end_year - start_year + 1})")
        logger.info(f"{'='*60}")
        success = build_cache(year, force=force)
        results[year] = success
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Cache Build Summary")
    logger.info(f"{'='*60}")
    success_count = sum(1 for v in results.values() if v)
    logger.info(f"Successfully cached: {success_count}/{len(results)} years")
    
    failed_years = [year for year, success in results.items() if not success]
    if failed_years:
        logger.warning(f"Failed years: {failed_years}")
    
    return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Build PBP data cache")
    parser.add_argument('--start-year', type=int, default=START_YEAR,
                       help=f'Start year for cache build (default: {START_YEAR})')
    parser.add_argument('--end-year', type=int, default=END_YEAR,
                       help=f'End year for cache build (default: {END_YEAR})')
    parser.add_argument('--year', type=int,
                       help='Build cache for a single year')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild even if cache exists')
    
    args = parser.parse_args()
    
    if args.year:
        # Build single year
        logger.info(f"Building cache for {args.year}...")
        success = build_cache(args.year, force=args.force)
        if success:
            logger.info(f"Successfully built cache for {args.year}")
        else:
            logger.error(f"Failed to build cache for {args.year}")
    else:
        # Build range of years
        logger.info(f"Building cache for years {args.start_year}-{args.end_year}...")
        results = build_cache_range(args.start_year, args.end_year, force=args.force)
        
        # Print summary
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        logger.info(f"\nCompleted: {success_count}/{total_count} years cached successfully")


if __name__ == "__main__":
    main()
