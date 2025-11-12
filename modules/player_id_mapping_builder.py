"""
Player ID Mapping Builder

Creates a mapping table between different player ID systems used across NFL data sources.
Primary mapping: pfr_player_id (used in snap counts) <-> gsis_id (used throughout codebase)

Output: cache/player_id_mapping.parquet

Columns:
- gsis_id: GSIS ID (primary identifier in most modules)
- pfr_id: Pro Football Reference player ID (used in snap count data)
- full_name: Player full name
- position: Position
- season: Most recent season player appeared in roster

One-time computation: ~2-3 minutes
Reusable for all future dataset rebuilds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from modules.constants import CACHE_DIR
from modules.logger import get_logger

logger = get_logger(__name__)


def build_player_id_mapping(start_year: int = 2009, end_year: int = 2024) -> pl.DataFrame:
    """
    Build player ID mapping from roster files.

    Args:
        start_year: First year to include
        end_year: Last year to include (inclusive)

    Returns:
        DataFrame with gsis_id, pfr_id, full_name, position, season
    """
    print("="*80)
    print("BUILDING PLAYER ID MAPPING TABLE")
    print("="*80)
    print(f"Years: {start_year}-{end_year}")

    # Step 1: Load all roster files
    print("\nStep 1: Loading roster files...")
    roster_list = []

    for year in range(start_year, end_year + 1):
        roster_file = Path(CACHE_DIR) / "rosters" / f"rosters-{year}.csv"
        if roster_file.exists():
            try:
                year_roster = pl.read_csv(roster_file)

                # Select only needed columns
                year_roster = year_roster.select([
                    'season',
                    'gsis_id',
                    'pfr_id',
                    'full_name',
                    'position'
                ])

                roster_list.append(year_roster)
                print(f"  {year}: {len(year_roster):,} roster entries")
            except Exception as e:
                print(f"  Error loading {year}: {e}")
        else:
            print(f"  {year}: File not found")

    if len(roster_list) == 0:
        raise ValueError("No roster data found!")

    # Step 2: Concatenate all rosters
    print(f"\nStep 2: Concatenating {len(roster_list)} years of roster data...")
    all_rosters = pl.concat(roster_list, how="diagonal")
    print(f"Total roster entries: {len(all_rosters):,}")

    # Step 3: Filter to entries with both IDs
    print("\nStep 3: Filtering to entries with both gsis_id and pfr_id...")
    valid_mappings = all_rosters.filter(
        pl.col('gsis_id').is_not_null() &
        pl.col('pfr_id').is_not_null() &
        (pl.col('gsis_id') != '') &
        (pl.col('pfr_id') != '')
    )
    print(f"Entries with both IDs: {len(valid_mappings):,} ({len(valid_mappings)/len(all_rosters)*100:.1f}%)")

    # Step 4: Deduplicate - keep most recent season for each pfr_id
    # This handles cases where player IDs change or players appear in multiple years
    print("\nStep 4: Deduplicating (keeping most recent season)...")

    # Sort by season descending, then take first of each pfr_id
    mapping = valid_mappings.sort('season', descending=True).group_by('pfr_id').first()

    print(f"Unique player mappings: {len(mapping):,}")

    # Step 5: Validate no duplicate gsis_ids
    gsis_counts = mapping.group_by('gsis_id').agg(pl.count().alias('count'))
    duplicates = gsis_counts.filter(pl.col('count') > 1)

    if len(duplicates) > 0:
        print(f"\nWARNING: Found {len(duplicates)} duplicate gsis_ids")
        print("This can happen if a player has multiple pfr_ids (team changes, name changes, etc.)")
        print("Using most recent mapping for each pfr_id")

    # Step 6: Sort and select final columns
    mapping = mapping.select([
        'gsis_id',
        'pfr_id',
        'full_name',
        'position',
        'season'
    ]).sort('gsis_id')

    return mapping


def save_mapping(mapping: pl.DataFrame, output_path: Path) -> None:
    """Save mapping to parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_parquet(output_path)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")


def print_summary(mapping: pl.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total player mappings: {len(mapping):,}")

    # Position breakdown
    position_counts = mapping.group_by('position').agg(
        pl.count().alias('count')
    ).sort('count', descending=True)

    print("\nMappings by position:")
    for row in position_counts.iter_rows(named=True):
        print(f"  {row['position']}: {row['count']:,}")

    # Year range
    min_year = mapping['season'].min()
    max_year = mapping['season'].max()
    print(f"\nYear range: {min_year}-{max_year}")

    print("\n" + "="*80)
    print("PLAYER ID MAPPING BUILD COMPLETE")
    print("="*80)
    print("\nNext step: Integrate into ml_training_data_builder.py multi-year cache")


def main():
    """Build and save player ID mapping."""
    # Build mapping
    mapping = build_player_id_mapping(start_year=2009, end_year=2024)

    # Save to cache
    output_file = Path(CACHE_DIR) / "player_id_mapping.parquet"
    save_mapping(mapping, output_file)

    # Print summary
    print_summary(mapping)


if __name__ == "__main__":
    main()
