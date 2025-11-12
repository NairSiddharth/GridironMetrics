"""
Analyze which of the 8 new features (7 game context + 1 injury) are helping vs hurting WR model.
Compare feature importance and correlations for the features added to align with QB/RB models.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import numpy as np
from modules.constants import CACHE_DIR
from modules.ml_ensemble import PropEnsembleModel

print("="*80)
print("FEATURE IMPACT ANALYSIS: 8 New Features")
print("="*80)

# Load the dataset
data_path = Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2009_2024.parquet"
df = pl.read_parquet(data_path)

# The 8 features we added
new_features = [
    'is_home',           # Game context
    'is_dome',           # Game context
    'game_temp',         # Game context (weather)
    'game_wind',         # Game context (weather)
    'vegas_total',       # Game context (betting)
    'vegas_spread',      # Game context (betting)
    'injury_status_score',  # Injury (from RB)
]

print(f"\nDataset: {len(df)} total examples (2009-2024, excluding 2020)")
print(f"Test set (2024): {len(df.filter(pl.col('year') == 2024))} examples")
print(f"Training set (2009-2023): {len(df.filter(pl.col('year') != 2024))} examples")

# Split into train/test
train_df = df.filter(pl.col('year') != 2024)
test_df = df.filter(pl.col('year') == 2024)

print("\n" + "="*80)
print("DATA QUALITY: New Features")
print("="*80)

for feature in new_features:
    # Check null/NaN counts
    null_count = train_df[feature].null_count()
    if train_df[feature].dtype in [pl.Float32, pl.Float64]:
        nan_count = train_df.filter(pl.col(feature).is_nan()).height
    else:
        nan_count = 0

    total_invalid = null_count + nan_count
    valid_count = len(train_df) - total_invalid
    valid_pct = (valid_count / len(train_df)) * 100

    print(f"{feature:<25} {valid_count:>6}/{len(train_df)} ({valid_pct:>5.1f}%) valid")

print("\n" + "="*80)
print("CORRELATIONS WITH TARGET (Training Set)")
print("="*80)

correlations = []
for feature in new_features:
    # Calculate correlation on non-null values
    valid_data = train_df.select([feature, 'target']).drop_nulls()
    if len(valid_data) > 0:
        corr = valid_data.select(pl.corr(feature, 'target'))[0, 0]
        if corr is None or np.isnan(corr):
            corr_str = "NaN"
        else:
            corr_str = f"{corr:>7.4f}"

        non_null_pct = (len(valid_data) / len(train_df)) * 100
        print(f"{feature:<25} {corr_str}  (on {non_null_pct:>5.1f}% of data)")

        if corr is not None and not np.isnan(corr):
            correlations.append((feature, abs(corr)))
    else:
        print(f"{feature:<25} NO VALID DATA")

print("\n" + "="*80)
print("FEATURE IMPORTANCE (from trained ensemble)")
print("="*80)

# Load the trained model
import pickle
model_path = Path(CACHE_DIR) / "ml_models" / "receiving_yards_wr_ensemble.pkl"
with open(model_path, 'rb') as f:
    ensemble = pickle.load(f)

# Get feature importance from tree models
print("\nXGBoost feature importance:")
xgb_importance = ensemble.models['xgboost'].named_steps['model'].feature_importances_
feature_names = ensemble.feature_names

# Map to feature names
importance_dict = dict(zip(feature_names, xgb_importance))

# Show importance for new features
for feature in new_features:
    if feature in importance_dict:
        importance = importance_dict[feature]
        print(f"  {feature:<25} {importance:>8.4f}")
    else:
        print(f"  {feature:<25} NOT IN MODEL")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\nBased on correlation and importance analysis:")
print("\n1. WEAK/HARMFUL FEATURES (consider removing):")
weak_features = []
for feature in new_features:
    if feature in importance_dict and importance_dict[feature] < 0.01:
        weak_features.append(feature)
        print(f"   - {feature}: Very low importance ({importance_dict[feature]:.4f})")

print("\n2. POTENTIALLY USEFUL FEATURES:")
for feature in new_features:
    if feature in importance_dict and importance_dict[feature] >= 0.01:
        print(f"   - {feature}: Importance {importance_dict[feature]:.4f}")

print("\n3. DATA QUALITY ISSUES:")
for feature in new_features:
    null_count = train_df[feature].null_count()
    if train_df[feature].dtype in [pl.Float32, pl.Float64]:
        nan_count = train_df.filter(pl.col(feature).is_nan()).height
    else:
        nan_count = 0

    total_invalid = null_count + nan_count
    if total_invalid > len(train_df) * 0.2:  # >20% missing
        valid_pct = ((len(train_df) - total_invalid) / len(train_df)) * 100
        print(f"   - {feature}: Only {valid_pct:.1f}% populated")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Try removing vegas features (vegas_total, vegas_spread)")
print("   - They have 27% missing data")
print("   - May not be predictive for individual WR performance")
print("\n2. Consider reverting to 2016-2024 year range")
print("   - 2009-2015 data may have different dynamics")
print("   - Compare model performance on different year ranges")
print("\n3. Try training with ONLY the baseline 28 features")
print("   - Establish if new features are helping or hurting")
print("\n4. Feature ablation study")
print("   - Remove features one at a time to find culprit")
