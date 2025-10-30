"""
nextgen_cache_builder.py

Builds and maintains a cache of NextGen Stats data (GPS tracking metrics)
from nflreadpy. Data includes separation at catch, cushion at snap, etc.
"""

from pathlib import Path
import nflreadpy as nfl
from modules.constants import CACHE_DIR
import polars as pl
from modules.logger import get_logger

logger = get_logger(__name__)

# NextGen Stats available starting 2016 (separation data best from 2017+)
NEXTGEN_START_YEAR = 2016

# Use absolute path for cache directory
CACHE_DIR = Path(CACHE_DIR).resolve() / "nextgen"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def build_nextgen_cache_for_year(year: int) -> bool:
    """
    Build NextGen Stats cache for a single year.
    
    Focuses on receiving stats (separation, cushion) for WR/TE/RB.
    
    Args:
        year: Season year to cache
        
    Returns:
        True if successful, False otherwise
    """
    if year < NEXTGEN_START_YEAR:
        logger.info(f"NextGen Stats not available for {year} (starts in {NEXTGEN_START_YEAR})")
        return False
    
    logger.info(f"Fetching NextGen Stats for {year}...")
    
    try:
        # Load receiving stats (has separation/cushion)
        ngs = nfl.load_nextgen_stats(stat_type='receiving', seasons=year)
    except Exception as e:
        logger.error(f"Error fetching NextGen Stats for {year}: {str(e)}")
        return False
    
    if ngs is None or (hasattr(ngs, 'is_empty') and ngs.is_empty()):
        logger.warning(f"No NextGen Stats returned for {year}")
        return False
    
    # Select columns we need
    columns_to_keep = [
        'season',
        'week',
        'player_display_name',
        'player_gsis_id',  # For joining with PBP
        'player_position',
        'team_abbr',
        'avg_cushion',  # Yards of cushion at snap
        'avg_separation',  # Yards of separation at catch
        'targets',
        'receptions'
    ]
    
    # Filter to columns that exist
    available_cols = [col for col in columns_to_keep if col in ngs.columns]
    ngs_subset = ngs.select(available_cols)
    
    # Filter to skill positions (WR, TE, RB)
    ngs_subset = ngs_subset.filter(
        pl.col('player_position').is_in(['WR', 'TE', 'RB'])
    )
    
    logger.info(f"Processing {len(ngs_subset)} NextGen receiver records for {year}")
    
    # Save as parquet
    out_path = CACHE_DIR / f"nextgen_{year}.parquet"
    try:
        ngs_subset.write_parquet(out_path)
        logger.info(f"Saved {out_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {out_path}: {str(e)}")
        return False


def load_nextgen_cache(year: int) -> pl.DataFrame | None:
    """
    Load NextGen Stats cache for a year.
    
    Args:
        year: Season year
        
    Returns:
        DataFrame with NextGen Stats or None if not available
    """
    if year < NEXTGEN_START_YEAR:
        return None
    
    cache_path = CACHE_DIR / f"nextgen_{year}.parquet"
    if not cache_path.exists():
        logger.warning(f"NextGen cache not found for {year}")
        return None
    
    try:
        return pl.read_parquet(cache_path)
    except Exception as e:
        logger.error(f"Error loading NextGen cache for {year}: {e}")
        return None


def build_full_nextgen_cache(start_year: int = NEXTGEN_START_YEAR, end_year: int = 2025) -> None:
    """
    Build NextGen Stats cache for multiple years.
    
    Args:
        start_year: First year to cache (default 2016)
        end_year: Last year to cache (default 2025)
    """
    logger.info(f"Building NextGen Stats cache for {start_year}-{end_year}")
    
    for year in range(start_year, end_year + 1):
        cache_path = CACHE_DIR / f"nextgen_{year}.parquet"
        if cache_path.exists():
            logger.info(f"NextGen cache already exists for {year}, skipping")
            continue
        
        success = build_nextgen_cache_for_year(year)
        if success:
            logger.info(f"✓ Cached NextGen Stats for {year}")
        else:
            logger.warning(f"✗ Failed to cache NextGen Stats for {year}")


if __name__ == "__main__":
    # Build cache for all available years
    build_full_nextgen_cache()
