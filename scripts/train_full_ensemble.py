"""
Train Ensemble on Full Historical Data (2015-2023)

Loads the complete training dataset and trains the 4-model ensemble.
This should significantly improve performance over the 2023-only model.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.ml_ensemble import PropEnsembleModel
from modules.constants import CACHE_DIR
import polars as pl
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("\n" + "="*60)
print("Full Ensemble Training (2015-2023)")
print("="*60)

# Load full training data
training_data_path = Path(CACHE_DIR) / "ml_training_data" / "passing_yards_2015_2023.parquet"

if not training_data_path.exists():
    print(f"\nERROR: Training data not found at {training_data_path}")
    print("Please run generate_full_training_data.py first")
    exit(1)

print(f"\nLoading training data from {training_data_path}...")
train_df = pl.read_parquet(training_data_path)

print(f"\n{'='*60}")
print("DATASET SUMMARY")
print(f"{'='*60}")
print(f"Total examples: {len(train_df):,}")
print(f"Features: {len(train_df.columns) - 3}")  # Exclude target, player_id, year
print(f"Years: 2015-2023")

# Year distribution
print(f"\n{'='*60}")
print("EXAMPLES PER YEAR")
print(f"{'='*60}")
year_counts = train_df.group_by('year').agg(pl.count()).sort('year')
for row in year_counts.iter_rows(named=True):
    print(f"  {row['year']}: {row['count']:,} examples")

print(f"\n{'='*60}")
print("TARGET STATISTICS")
print(f"{'='*60}")
print(f"Mean: {train_df['target'].mean():.2f} yards")
print(f"Std: {train_df['target'].std():.2f} yards")
print(f"Min: {train_df['target'].min():.2f} yards")
print(f"Max: {train_df['target'].max():.2f} yards")

# Initialize and train ensemble
print(f"\n{'='*60}")
print("TRAINING ENSEMBLE")
print(f"{'='*60}")

ensemble = PropEnsembleModel(prop_type='passing_yards')
performance = ensemble.train(train_df, n_splits=5, verbose=True)

# Save model
ensemble.save()

print(f"\n{'='*60}")
print("PERFORMANCE COMPARISON")
print(f"{'='*60}")
print("\nBaseline (2023 only, 551 examples):")
print("  Random Forest: MAE = 70.06 ± 8.19 yards")
print("  LightGBM:      MAE = 71.91 ± 7.51 yards")
print("  XGBoost:       MAE = 73.46 ± 8.18 yards")
print("  CatBoost:      MAE = 73.65 ± 6.64 yards")

print(f"\nFull Model (2015-2023, {len(train_df):,} examples):")
for model_name, metrics in performance.items():
    print(f"  {model_name:15s}: MAE = {metrics['mae_mean']:.2f} ± {metrics['mae_std']:.2f} yards")

# Calculate improvement
baseline_mae = 70.06  # Best model from 2023 training
full_mae = min([m['mae_mean'] for m in performance.values()])
improvement = ((baseline_mae - full_mae) / baseline_mae) * 100

print(f"\n{'='*60}")
print(f"Best model MAE improvement: {improvement:.1f}%")
print(f"Baseline: {baseline_mae:.2f} yards → Full: {full_mae:.2f} yards")
print(f"{'='*60}")

print("\n" + "="*60)
print("Ensemble trained and saved successfully")
print("="*60)
