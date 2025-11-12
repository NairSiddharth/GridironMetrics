"""
snap_count_cache_builder.py

Builds and maintains a cache of player snap counts from nflreadpy.
Used to calculate route participation % and targets per route run (TPRR).
"""

from pathlib import Path
import nflreadpy as nfl
from modules.constants import CACHE_DIR
import polars as pl
from modules.logger import get_logger

logger = get_logger(__name__)

# Snap counts available starting 2012, but we'll focus on 2016+ for consistency with NextGen
SNAP_COUNT_START_YEAR = 2016

# Use absolute path for cache directory
CACHE_DIR = Path(CACHE_DIR).resolve() / "snap_counts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def build_snap_count_cache_for_year(year: int) -> bool:
    """
    Build snap count cache for a single year.

    Snap counts show how many plays a player was on the field.
    For WRs, we approximate route participation as offense_snaps on passing plays.

    Args:
        year: Season year to cache

    Returns:
        True if successful, False otherwise
    """
    if year < SNAP_COUNT_START_YEAR:
        logger.info(f"Snap counts not cached for {year} (caching starts in {SNAP_COUNT_START_YEAR})")
        return False

    logger.info(f"Fetching snap counts for {year}...")

    try:
        # Load snap counts from nflreadpy
        snaps = nfl.load_snap_counts(seasons=year)
    except Exception as e:
        logger.error(f"Error fetching snap counts for {year}: {str(e)}")
        return False

    if snaps is None or (hasattr(snaps, 'is_empty') and snaps.is_empty()):
        logger.warning(f"No snap counts returned for {year}")
        return False

    # Select columns we need
    columns_to_keep = [
        'season',
        'game_id',
        'week',
        'player',  # Display name
        'pfr_player_id',  # For joining
        'position',
        'team',
        'opponent',
        'offense_snaps',  # Total offensive snaps
        'offense_pct',  # % of team's offensive snaps
        'defense_snaps',  # Defensive snaps (0 for offensive players)
        'defense_pct',  # % of team's defensive snaps
        'st_snaps',  # Special teams snaps
        'st_pct'  # % of special teams snaps
    ]

    # Filter to columns that exist
    available_cols = [col for col in columns_to_keep if col in snaps.columns]
    snaps_subset = snaps.select(available_cols)

    # Filter to skill positions (WR, TE, RB) - these are who we care about for receiving
    skill_positions = ['WR', 'TE', 'RB']
    snaps_subset = snaps_subset.filter(
        pl.col('position').is_in(skill_positions)
    )

    logger.info(f"Processing {len(snaps_subset)} snap count records for {year}")

    # Save as parquet
    out_path = CACHE_DIR / f"snap_counts_{year}.parquet"
    try:
        snaps_subset.write_parquet(out_path)
        logger.info(f"Snap counts saved to: {out_path}")
        logger.info(f"  Records: {len(snaps_subset):,}")
        logger.info(f"  Positions: {snaps_subset['position'].value_counts().to_dict()}")
        return True
    except Exception as e:
        logger.error(f"Error saving snap counts: {str(e)}")
        return False


def load_snap_count_cache(year: int):
    """
    Load snap count cache for a year.

    Args:
        year: Season year

    Returns:
        Polars DataFrame or None if not cached
    """
    cache_file = CACHE_DIR / f"snap_counts_{year}.parquet"

    if not cache_file.exists():
        logger.debug(f"Snap count cache not found for {year}: {cache_file}")
        return None

    try:
        return pl.read_parquet(cache_file)
    except Exception as e:
        logger.warning(f"Error loading snap count cache for {year}: {e}")
        return None


def build_all_snap_count_caches(start_year: int = 2016, end_year: int = 2024):
    """
    Build snap count caches for all years.

    Args:
        start_year: First year to cache (default: 2016)
        end_year: Last year to cache (default: 2024)
    """
    logger.info("="*60)
    logger.info(f"Building Snap Count Caches ({start_year}-{end_year})")
    logger.info("="*60)

    success_count = 0
    fail_count = 0

    for year in range(start_year, end_year + 1):
        if build_snap_count_cache_for_year(year):
            success_count += 1
        else:
            fail_count += 1

    logger.info("")
    logger.info("="*60)
    logger.info(f"Snap Count Cache Build Complete")
    logger.info("="*60)
    logger.info(f"  Successful: {success_count} years")
    logger.info(f"  Failed: {fail_count} years")
    logger.info(f"  Cache directory: {CACHE_DIR}")


if __name__ == "__main__":
    # Build caches for all years
    build_all_snap_count_caches(start_year=2016, end_year=2024)
