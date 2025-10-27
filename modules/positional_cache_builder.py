"""
nfl_cache_builder.py

Builds and maintains a cache of NFL weekly player statistics (regular season only)
from nflreadpy. Data is split by position, year, and saved as CSVs in the 'cache' folder.
"""

from pathlib import Path
import nflreadpy as nfl
from constants import START_YEAR, END_YEAR, CACHE_DIR
import polars as pl

CACHE_DIR = Path(CACHE_DIR)

def build_cache_for_year(year: int):
    print(f"Fetching weekly player stats for {year}...")
    df = nfl.load_player_stats(seasons=year, summary_level="week")
    df = df.filter(pl.col("season_type") == "REG")

    positions = df["position"].drop_nulls().unique().to_list()
    for pos in positions:
        pos_slug = pos.lower()
        folder = CACHE_DIR / pos_slug
        folder.mkdir(parents=True, exist_ok=True)

        pos_df = df.filter(pl.col("position") == pos)
        pos_df = pos_df.sort(["player_id", "week"])

        out_path = folder / f"{pos_slug}-{year}.csv"
        pos_df.write_csv(out_path)
        print(f"Saved {out_path}")


def build_full_cache(start_year=START_YEAR, end_year=END_YEAR):
    for year in range(start_year, end_year + 1):
        build_cache_for_year(year)
    print("All caches built successfully.")


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
    print("Checking cache consistency...")
    missing = cache_is_up_to_date(start_year, end_year)
    if not missing:
        print("Cache is complete and up to date.")
        return

    if ("ALL", "NONE") in missing:
        print("Cache directory not found — rebuilding full cache.")
        build_full_cache(start_year, end_year)
        return

    years_needed = sorted(set([year for year, _ in missing]))
    print(f"Missing cache files for: {missing}")
    for year in years_needed:
        build_cache_for_year(year)

    print("Cache now up to date.")


if __name__ == "__main__":
    ensure_cache_up_to_date()
