"""
ftn_cache_builder.py

Builds and maintains a cache of FTN charting data (human-charted play characteristics)
from nflreadpy. Data includes play action, RPO, blitz count, contested catches, etc.
"""

from pathlib import Path
import nflreadpy as nfl
from modules.constants import CACHE_DIR
import polars as pl
from modules.logger import get_logger

logger = get_logger(__name__)

# FTN data available starting 2022
FTN_START_YEAR = 2022

# Use absolute path for cache directory
CACHE_DIR = Path(CACHE_DIR).resolve() / "ftn"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def build_ftn_cache_for_year(year: int) -> bool:
    """
    Build FTN charting cache for a single year.
    
    Args:
        year: Season year to cache
        
    Returns:
        True if successful, False otherwise
    """
    if year < FTN_START_YEAR:
        logger.info(f"FTN data not available for {year} (starts in {FTN_START_YEAR})")
        return False
    
    logger.info(f"Fetching FTN charting data for {year}...")
    
    try:
        ftn = nfl.load_ftn_charting(seasons=year)
    except Exception as e:
        logger.error(f"Error fetching FTN data for {year}: {str(e)}")
        return False
    
    if ftn is None or (hasattr(ftn, 'is_empty') and ftn.is_empty()):
        logger.warning(f"No FTN data returned for {year}")
        return False
    
    # Select only the columns we need
    columns_to_keep = [
        'nflverse_game_id',
        'nflverse_play_id',
        # QB flags
        'is_play_action',
        'is_qb_out_of_pocket',
        'n_blitzers',
        # WR/TE flags
        'is_contested_ball',
        'is_drop',
        # All positions
        'is_rpo',
        'is_screen_pass',
        'n_defense_box'  # More accurate than inferred defenders_in_box
    ]
    
    ftn_subset = ftn.select(columns_to_keep)
    
    logger.info(f"Processing {len(ftn_subset)} FTN charted plays for {year}")
    
    # Save as parquet for efficient joining with PBP
    out_path = CACHE_DIR / f"ftn_{year}.parquet"
    try:
        ftn_subset.write_parquet(out_path)
        logger.info(f"Saved {out_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {out_path}: {str(e)}")
        return False


def build_full_ftn_cache(start_year: int = FTN_START_YEAR, end_year: int = 2025) -> None:
    """Build FTN cache for all available years."""
    logger.info(f"Building FTN cache for years {start_year}-{end_year}")
    
    success_count = 0
    for year in range(start_year, end_year + 1):
        if build_ftn_cache_for_year(year):
            success_count += 1
    
    logger.info(f"Successfully built FTN cache for {success_count} years")


def ftn_cache_exists(year: int) -> bool:
    """Check if FTN cache exists for a given year."""
    if year < FTN_START_YEAR:
        return False  # Data not available
    
    cache_path = CACHE_DIR / f"ftn_{year}.parquet"
    return cache_path.exists()


def load_ftn_cache(year: int) -> pl.DataFrame:
    """
    Load FTN cache for a given year.
    
    Args:
        year: Season year to load
        
    Returns:
        DataFrame with FTN charting data, or None if not available
    """
    if not ftn_cache_exists(year):
        logger.debug(f"FTN cache not available for {year}")
        return None
    
    cache_path = CACHE_DIR / f"ftn_{year}.parquet"
    try:
        return pl.read_parquet(cache_path)
    except Exception as e:
        logger.error(f"Error loading FTN cache for {year}: {str(e)}")
        return None


if __name__ == "__main__":
    # Build cache for all available years
    build_full_ftn_cache()
