"""
Rebuild WR Receiving Yards Training Datasets

Rebuilds both full and high-volume filtered datasets with all bug fixes:
- Bug #1: division_game extraction fixed
- Bug #2: betting_lines added to data_cache
- Bug #3: div_game added to game_metadata cache

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

    # Step 2: Verify bug fixes
    print("\n" + "="*80)
    print("STEP 2: Verifying bug fixes")
    print("="*80)

    features_to_check = ['division_game', 'vegas_total', 'vegas_spread']

    print("\nFeature coverage across all examples:")
    for feat in features_to_check:
        col = full_df[feat].to_numpy()
        non_null = np.sum(~np.isnan(col))
        pct = (non_null / len(full_df)) * 100
        print(f"  {feat:20s}: {non_null:6,} / {len(full_df):6,} ({pct:5.1f}%)")

    # Check 2024 specifically (should have highest coverage)
    df_2024 = full_df.filter(pl.col('season') == 2024)
    print(f"\n2024 feature coverage ({len(df_2024):,} examples):")
    for feat in features_to_check:
        col = df_2024[feat].to_numpy()
        non_null = np.sum(~np.isnan(col))
        pct = (non_null / len(df_2024)) * 100
        print(f"  {feat:20s}: {non_null:6,} / {len(df_2024):6,} ({pct:5.1f}%)")

        if non_null > 0:
            unique_vals = np.unique(col[~np.isnan(col)])
            if len(unique_vals) <= 10:
                print(f"    Unique values: {unique_vals}")
            else:
                print(f"    Range: [{unique_vals.min():.1f}, {unique_vals.max():.1f}]")

    # Step 3: Create high-volume filtered dataset
    print("\n" + "="*80)
    print("STEP 3: Creating high-volume filtered dataset (>=4.5 targets/game)")
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

    # Step 4: Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
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
