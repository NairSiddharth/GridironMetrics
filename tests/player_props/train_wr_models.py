"""
Train Full and High-Volume WR Models

Trains both models and compares performance on 2024 real betting lines.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
from modules.ml_ensemble import PropEnsembleModel
from modules.constants import CACHE_DIR

def main():
    print("="*80)
    print("TRAINING WR RECEIVING YARDS MODELS")
    print("="*80)

    # Train full dataset model
    print("\n" + "="*80)
    print("STEP 1: Training FULL DATASET MODEL")
    print("="*80)

    full_df = pl.read_parquet(Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2009_2024.parquet")
    print(f"Loaded full dataset: {len(full_df):,} examples, {len(full_df.columns)} features")

    # CRITICAL: Exclude 2024 from training to prevent data leakage
    # 2024 is our holdout test set
    train_df_full = full_df.filter(pl.col('year') < 2024)
    print(f"Training dataset (2009-2023): {len(train_df_full):,} examples")
    print(f"Holdout test set (2024): {len(full_df.filter(pl.col('year') == 2024)):,} examples")

    model_full = PropEnsembleModel(prop_type='receiving_yards_wr')
    model_full.train(train_df_full, optimize_hyperparams=True, n_iter=3)

    model_path_full = Path(CACHE_DIR) / "ml_models" / "receiving_yards_wr_ensemble.pkl"
    model_path_full.parent.mkdir(parents=True, exist_ok=True)
    model_full.save(str(model_path_full))
    print(f"\nSaved full model to: {model_path_full}")

    # Train high-volume model
    print("\n" + "="*80)
    print("STEP 2: Training HIGH-VOLUME MODEL (>=4.5 targets/game)")
    print("="*80)

    high_vol_df = pl.read_parquet(Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2009_2024_high_volume.parquet")
    print(f"Loaded high-volume dataset: {len(high_vol_df):,} examples, {len(high_vol_df.columns)} features")

    # CRITICAL: Exclude 2024 from training to prevent data leakage
    train_df_high_vol = high_vol_df.filter(pl.col('year') < 2024)
    print(f"Training dataset (2009-2023): {len(train_df_high_vol):,} examples")
    print(f"Holdout test set (2024): {len(high_vol_df.filter(pl.col('year') == 2024)):,} examples")

    model_high_vol = PropEnsembleModel(prop_type='receiving_yards_wr')
    model_high_vol.train(train_df_high_vol, optimize_hyperparams=True, n_iter=3)

    model_path_high_vol = Path(CACHE_DIR) / "ml_models" / "receiving_yards_wr_ensemble_high_volume.pkl"
    model_high_vol.save(str(model_path_high_vol))
    print(f"\nSaved high-volume model to: {model_path_high_vol}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nNext step: Compare models on 2024 real betting lines")
    print("Run: python tests/player_props/compare_models_real_lines.py")

if __name__ == "__main__":
    main()
