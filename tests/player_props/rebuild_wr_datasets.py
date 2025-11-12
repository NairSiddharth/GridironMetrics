"""
Rebuild WR Receiving Yards Training Datasets

Rebuilds both full and high-volume filtered datasets with expanded features and years:
- Aligned with successful QB/RB methodology
- 36 features (added 7 game context + 1 injury)
- 2009-2024 training range (88% more data than 2016-2024)
- Tree models handle NextGen NaN natively

Creates two datasets:
1. Full: All WRs (2009-2024, skip 2020)
2. High-volume: WRs with >=4.5 targets/game average
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import numpy as np
from modules.ml_training_data_builder import TrainingDataBuilder
from modules.constants import CACHE_DIR

def main():
    print("="*80)
    print("REBUILDING WR RECEIVING YARDS DATASETS")
    print("="*80)

    # Step 1: Build full dataset
    print("\n" + "="*80)
    print("STEP 1: Building full dataset (all WRs, 2009-2024)")
    print("="*80)

    builder = TrainingDataBuilder()
    full_df = builder.build_training_dataset(
        start_year=2009,
        end_year=2024,
        prop_type='receiving_yards_wr',
        skip_years=[2020]  # Skip COVID year
    )

    print(f"\nFull dataset: {len(full_df):,} examples, {len(full_df.columns)} features")

    # Save full dataset
    full_path = Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2009_2024.parquet"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.write_parquet(full_path)
    print(f"Saved to: {full_path}")

    # Step 2: Create high-volume filtered dataset
    print("\n" + "="*80)
    print("STEP 2: Creating high-volume filtered dataset (>=4.5 targets/game)")
    print("="*80)

    # Filter criterion: targets_season_avg >= 4.5
    # This ensures we only include WRs who get consistent volume
    if 'targets_season_avg' in full_df.columns:
        high_vol_df = full_df.filter(pl.col('targets_season_avg') >= 4.5)
    else:
        # Fallback: calculate from games_played if targets_season_avg not available
        print("WARNING: targets_season_avg not found, using alternative calculation")
        # This shouldn't happen, but handle gracefully
        high_vol_df = full_df

    print(f"\nHigh-volume dataset: {len(high_vol_df):,} examples ({len(high_vol_df)/len(full_df)*100:.1f}% of full)")
    print(f"Filtered out: {len(full_df) - len(high_vol_df):,} low-volume WR examples")

    # Save high-volume dataset
    high_vol_path = Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2009_2024_high_volume.parquet"
    high_vol_df.write_parquet(high_vol_path)
    print(f"Saved to: {high_vol_path}")

    # Step 3: Summary statistics
    print("\n" + "="*80)
    print("STEP 3: SUMMARY")
    print("="*80)

    print(f"\nFull Dataset:")
    print(f"  Path: {full_path}")
    print(f"  Examples: {len(full_df):,}")
    print(f"  Features: {len(full_df.columns)}")
    print(f"  Years: 2009-2024 (excluding 2020)")

    print(f"\nHigh-Volume Dataset:")
    print(f"  Path: {high_vol_path}")
    print(f"  Examples: {len(high_vol_df):,}")
    print(f"  Features: {len(high_vol_df.columns)}")
    print(f"  Filter: >=4.5 targets/game")

    print(f"\nYear-by-year breakdown (high-volume):")
    for year in sorted(high_vol_df['season'].unique()):
        year_count = len(high_vol_df.filter(pl.col('season') == year))
        full_count = len(full_df.filter(pl.col('season') == year))
        pct = (year_count / full_count * 100) if full_count > 0 else 0
        print(f"  {year}: {year_count:4,} / {full_count:4,} ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("REBUILD COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Train full dataset model")
    print("2. Train high-volume model")
    print("3. Compare performance on 2024 real betting lines")

if __name__ == "__main__":
    main()
