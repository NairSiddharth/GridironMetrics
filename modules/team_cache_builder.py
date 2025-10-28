"""
team_cache_builder.py

Builds and maintains a cache of NFL weekly team statistics (regular season only)
from nflreadpy. Data is split by position, year, and saved as CSVs in the 'cache' folder.
"""

from pathlib import Path
import nflreadpy as nfl
from constants import START_YEAR, END_YEAR, CACHE_DIR
import polars as pl
from logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(CACHE_DIR) / "team_stats"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def build_team_cache_for_year(year: int):
    logger.info(f"Fetching weekly team stats for {year}...")
    try:
        df = nfl.load_team_stats(seasons=year, summary_level="week")
    except Exception as e:
        logger.error(f"Error fetching data for year {year}: {str(e)}")
        return

    # Debug: Check if DataFrame is empty or None
    if df is None or (hasattr(df, 'is_empty') and df.is_empty()):
        logger.warning(f"No data returned for year {year}.")
        return

    # keep regular season only
    if "season_type" in df.columns:
        df = df.filter(pl.col("season_type") == "REG")
        logger.debug(f"Filtered to {len(df)} regular season rows")

    # determine a team identifier column (fallback list)
    cols = list(df.columns)
    team_candidates = [c for c in ["team", "team_abbr", "team_id", "team_name", "team_code"] if c in cols]
    if not team_candidates:
        logger.error(f"Available columns: {cols}")
        raise RuntimeError(f"No team identifier column found in DataFrame columns")
    team_col = team_candidates[0]
    logger.debug(f"Using {team_col} as team identifier")

    teams = df[team_col].drop_nulls().unique().to_list()
    logger.info(f"Found {len(teams)} teams: {teams}")  # Debug: List of teams found

    for team in teams:
        team_slug = str(team).lower().replace(" ", "_")
        folder = CACHE_DIR / team_slug
        folder.mkdir(parents=True, exist_ok=True)

        team_df = df.filter(pl.col(team_col) == team)
        # Debug: Check if team_df is empty
        if team_df.is_empty():
            logger.warning(f"No data for team {team} in year {year}.")
            continue

        # sort by week if available
        if "week" in team_df.columns:
            team_df = team_df.sort("week")

        out_path = folder / f"{team_slug}-{year}.csv"
        try:
            team_df.write_csv(out_path)
            logger.info(f"Saved {out_path}")
        except Exception as e:
            logger.error(f"Failed to save {out_path}: {str(e)}")


def build_full_cache(start_year=START_YEAR, end_year=END_YEAR):
    for year in range(start_year, end_year + 1):
        build_team_cache_for_year(year)
    logger.info("All team stats caches built successfully.")


def cache_is_up_to_date(start_year=START_YEAR, end_year=END_YEAR):
    # If the cache directory doesn't exist or is empty, trigger a full rebuild
    if not CACHE_DIR.exists() or not any(CACHE_DIR.iterdir()):
        return [("ALL", "NONE")]
    
    # For team stats, we need at least one team's data for each year
    missing = []
    for year in range(start_year, end_year + 1):
        # Check if we have any team data for this year
        year_has_data = False
        for team_dir in CACHE_DIR.iterdir():
            if team_dir.is_dir():
                year_file = team_dir / f"{team_dir.name}-{year}.csv"
                if year_file.exists():
                    year_has_data = True
                    break
        if not year_has_data:
            missing.append((year, "team_stats"))
    return missing


def ensure_cache_up_to_date(start_year=START_YEAR, end_year=END_YEAR):
    logger.info(f"Checking team stats cache consistency for years {start_year}-{end_year}...")
    missing = cache_is_up_to_date(start_year, end_year)
    
    if not missing:
        existing = [d.name for d in CACHE_DIR.iterdir() if d.is_dir()]
        logger.info(f"Cache is complete and up to date. Found {len(existing)} team directories.")
        return

    if ("ALL", "NONE") in missing:
        logger.warning("Cache is empty or missing — rebuilding full team stats cache.")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        build_full_cache(start_year, end_year)
        return

    years_needed = sorted(set([year for year, _ in missing]))
    logger.warning(f"Missing team stats for years: {years_needed}")
    for year in years_needed:
        build_team_cache_for_year(year)

    print("Cache now up to date.")


if __name__ == "__main__":
    ensure_cache_up_to_date()
