"""
positional_cache_builder.py

Builds and maintains a cache of NFL weekly player statistics (regular season only)
from nflreadpy. Data is split by position, year, and saved as CSVs in the 'cache' folder.
"""

from pathlib import Path
import nflreadpy as nfl
from constants import START_YEAR, END_YEAR, CACHE_DIR
import polars as pl
import logging
from logger import LOG_DIR  # This ensures logger is configured

CACHE_DIR = (Path(CACHE_DIR) / "positional_player_stats").mkdir(parents=True, exist_ok=True)

def build_positional_cache_for_year(year: int):
    logger = logging.getLogger(__name__)
    logger.info(f"Fetching weekly player stats for {year}...")
    
    try:
        df = nfl.load_player_stats(seasons=year, summary_level="week")
    except Exception as e:
        logger.error(f"Error fetching player stats for year {year}: {str(e)}")
        return

    # Debug: Check if DataFrame is empty or None
    if df is None or (hasattr(df, 'is_empty') and df.is_empty()):
        logger.warning(f"No player data returned for year {year}.")
        return

    # Keep regular season only
    df = df.filter(pl.col("season_type") == "REG")
    logger.debug(f"Filtered to {len(df)} regular season rows")

    positions = df["position"].drop_nulls().unique().to_list()
    logger.info(f"Found {len(positions)} positions: {positions}")

    for pos in positions:
        pos_slug = pos.lower()
        folder = CACHE_DIR / pos_slug
        folder.mkdir(parents=True, exist_ok=True)

        pos_df = df.filter(pl.col("position") == pos)
        pos_df = pos_df.sort(["player_id", "week"])
        player_count = len(pos_df["player_id"].unique())
        logger.debug(f"Processing {player_count} {pos} players for {year}")

        out_path = folder / f"{pos_slug}-{year}.csv"
        try:
            pos_df.write_csv(out_path)
            logger.info(f"Saved {out_path}")
        except Exception as e:
            logger.error(f"Failed to save {out_path}: {str(e)}")


def build_full_cache(start_year=START_YEAR, end_year=END_YEAR):
    logger = logging.getLogger(__name__)
    for year in range(start_year, end_year + 1):
        build_positional_cache_for_year(year)
    logger.info("All positional player stats caches built successfully.")


def cache_is_up_to_date(start_year=START_YEAR, end_year=END_YEAR):
    missing = []
    if not CACHE_DIR.exists():
        return [("ALL", "NONE")]

    positions = [d.name for d in CACHE_DIR.iterdir() if d.is_dir()]
    for year in range(start_year, end_year + 1):
        for pos in positions:
            expected_file = CACHE_DIR / pos / f"{pos}-{year}.csv"
            if not expected_file.exists():
                missing.append((year, pos))
    return missing


def ensure_cache_up_to_date(start_year=START_YEAR, end_year=END_YEAR):
    logger = logging.getLogger(__name__)
    logger.info(f"Checking positional player stats cache consistency for years {start_year}-{end_year}...")
    missing = cache_is_up_to_date(start_year, end_year)
    
    if not missing:
        existing = [d.name for d in CACHE_DIR.iterdir() if d.is_dir()]
        logger.info(f"Cache is complete and up to date. Found {len(existing)} position directories.")
        return

    if ("ALL", "NONE") in missing:
        logger.warning("Cache directory not found or empty — rebuilding full positional cache.")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        build_full_cache(start_year, end_year)
        return

    years_needed = sorted(set([year for year, _ in missing]))
    positions_needed = sorted(set([pos for _, pos in missing]))
    logger.warning(f"Missing data for years: {years_needed}")
    logger.debug(f"Affected positions: {positions_needed}")
    
    for year in years_needed:
        build_positional_cache_for_year(year)

    logger.info("Positional player stats cache now up to date.")


if __name__ == "__main__":
    ensure_cache_up_to_date()
