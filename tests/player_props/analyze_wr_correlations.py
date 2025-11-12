"""
WR Receiving Yards High-Volume Model - Feature Correlation Analysis

Analyzes correlations between features and target to understand:
1. Which features are most predictive
2. Which features are redundant (multicollinearity)
3. Which features are associated with prediction errors
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import numpy as np
from modules.ml_ensemble import PropEnsembleModel
from modules.constants import CACHE_DIR

def calculate_correlations(df: pl.DataFrame, features: list, target: str = 'target'):
    """Calculate Pearson correlation between features and target."""
    correlations = []

    for feature in features:
        # Get non-null pairs
        valid_mask = df[feature].is_not_null() & df[target].is_not_null()
        if valid_mask.sum() == 0:
            correlations.append((feature, 0.0, 0))
            continue

        feature_vals = df.filter(valid_mask)[feature].to_numpy()
        target_vals = df.filter(valid_mask)[target].to_numpy()

        if len(feature_vals) > 1 and np.std(feature_vals) > 0 and np.std(target_vals) > 0:
            corr = np.corrcoef(feature_vals, target_vals)[0, 1]
            correlations.append((feature, corr, len(feature_vals)))
        else:
            correlations.append((feature, 0.0, len(feature_vals)))

    return sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

def calculate_feature_correlations(df: pl.DataFrame, features: list):
    """Calculate correlation matrix between all features."""
    n = len(features)
    corr_matrix = np.zeros((n, n))

    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue

            valid_mask = df[feat1].is_not_null() & df[feat2].is_not_null()
            if valid_mask.sum() == 0:
                continue

            vals1 = df.filter(valid_mask)[feat1].to_numpy()
            vals2 = df.filter(valid_mask)[feat2].to_numpy()

            if len(vals1) > 1 and np.std(vals1) > 0 and np.std(vals2) > 0:
                corr_matrix[i, j] = np.corrcoef(vals1, vals2)[0, 1]

    return corr_matrix

def main():
    print("="*80)
    print("WR RECEIVING YARDS HIGH-VOLUME MODEL - CORRELATION ANALYSIS")
    print("="*80)

    # Load high-volume dataset
    dataset_path = Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2016_2024_high_volume.parquet"
    df = pl.read_parquet(dataset_path)

    print(f"\nDataset: {len(df):,} examples")
    print(f"Columns: {len(df.columns)}")

    # Get model features (exclude metadata)
    all_features = [col for col in df.columns if col not in ['target', 'player_id', 'year', 'week']]
    print(f"Features: {len(all_features)}")

    # Feature categories
    feature_categories = {
        'Baseline': ['weighted_avg', 'games_played'],
        'Efficiency': ['success_rate_3wk'],
        'Catch Rate': ['avg_target_depth', 'yac_pct'],
        'Volume': ['targets_season_avg', 'yards_per_target_season', 'yards_per_target_3wk'],
        'Volume Trends': ['target_trend_3wk', 'target_volatility_cv'],
        'QB Context': ['qb_adot_3wk', 'qb_pass_yards_3wk', 'qb_pass_attempts_3wk',
                       'qb_comp_pct_3wk', 'qb_ypa_3wk', 'qb_target_concentration'],
        'Opponent Defense': ['opp_def_pass_ypa', 'opp_def_pass_td_rate',
                            'opp_def_avg_depth_allowed', 'opp_def_pass_rank_pct'],
        'Matchup History': ['wr_vs_team_avg_yards_last3', 'wr_vs_team_receptions_last3'],
        'Weather': ['game_temp'],
        'Game Context': ['is_home', 'is_dome'],
        'NextGen': ['avg_separation'],
        'Route Participation': ['route_participation_pct', 'targets_per_route_run', 'route_target_gap']
    }

    # Flatten for full feature list
    categorized_features = []
    for category, feats in feature_categories.items():
        categorized_features.extend(feats)

    # Calculate feature-target correlations
    print("\n" + "="*80)
    print("FEATURE-TARGET CORRELATIONS")
    print("="*80)

    correlations = calculate_correlations(df, all_features, 'target')

    print(f"\n{'Rank':<6} {'Feature':<35} {'Correlation':<15} {'N':<10}")
    print("-"*80)
    for rank, (feature, corr, n) in enumerate(correlations, 1):
        print(f"{rank:<6} {feature:<35} {corr:>+.4f} {'':<6} {n:>8,}")

    # Strong correlations
    strong_corr = [(f, c, n) for f, c, n in correlations if abs(c) > 0.3]
    print(f"\nStrong correlations (|r| > 0.3): {len(strong_corr)}")
    for feat, corr, n in strong_corr:
        print(f"  {feat:<35} r={corr:+.4f}")

    # Descriptive stats for top 5
    print("\n" + "="*80)
    print("TOP 5 FEATURE STATISTICS")
    print("="*80)

    for rank, (feature, corr, n) in enumerate(correlations[:5], 1):
        feat_data = df[feature].drop_nulls()
        print(f"\n{rank}. {feature} (r={corr:+.4f})")
        print(f"   Mean:   {feat_data.mean():.2f}")
        print(f"   Std:    {feat_data.std():.2f}")
        print(f"   Min:    {feat_data.min():.2f}")
        print(f"   Max:    {feat_data.max():.2f}")
        print(f"   Median: {feat_data.median():.2f}")

    # Category analysis
    print("\n" + "="*80)
    print("CORRELATION BY CATEGORY")
    print("="*80)

    category_stats = []
    for category, feats in feature_categories.items():
        cat_corrs = [corr for feat, corr, n in correlations if feat in feats]
        if cat_corrs:
            avg_corr = np.mean([abs(c) for c in cat_corrs])
            max_corr = max(cat_corrs, key=abs)
            category_stats.append((category, avg_corr, max_corr, len(cat_corrs)))

    category_stats.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Category':<25} {'Avg |r|':<12} {'Best r':<12} {'# Features':<12}")
    print("-"*80)
    for category, avg_corr, max_corr, n_feats in category_stats:
        print(f"{category:<25} {avg_corr:>8.4f} {'':4} {max_corr:>+8.4f} {'':4} {n_feats:>8}")

    # Multicollinearity analysis
    print("\n" + "="*80)
    print("MULTICOLLINEARITY ANALYSIS")
    print("="*80)

    print("\nCalculating feature-feature correlation matrix...")
    corr_matrix = calculate_feature_correlations(df, all_features)

    # Find high correlations
    high_corr_pairs = []
    for i, feat1 in enumerate(all_features):
        for j, feat2 in enumerate(all_features):
            if i < j:  # Only upper triangle
                corr_val = corr_matrix[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((feat1, feat2, corr_val))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\nHigh correlations between features (|r| > 0.7): {len(high_corr_pairs)}")
    if high_corr_pairs:
        print(f"\n{'Feature 1':<35} {'Feature 2':<35} {'Correlation':<15}")
        print("-"*80)
        for feat1, feat2, corr in high_corr_pairs:
            print(f"{feat1:<35} {feat2:<35} {corr:>+.4f}")
    else:
        print("No feature pairs with |r| > 0.7 (good - low multicollinearity)")

    # Prediction error analysis
    print("\n" + "="*80)
    print("PREDICTION ERROR ANALYSIS")
    print("="*80)

    print("\nLoading high-volume model...")
    model = PropEnsembleModel(prop_type='receiving_yards_wr')
    model_path = Path(CACHE_DIR) / "ml_models" / "receiving_yards_wr_ensemble_high_volume.pkl"
    model.load(str(model_path))

    print("Generating predictions...")
    predictions = model.predict_batch(df)
    actuals = df['target'].to_numpy()

    # Calculate errors
    errors = actuals - predictions
    abs_errors = np.abs(errors)

    print(f"\nPrediction statistics:")
    print(f"  MAE:  {np.mean(abs_errors):.2f} yards")
    print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.2f} yards")
    print(f"  Mean error: {np.mean(errors):+.2f} yards (bias)")

    # Correlate features with absolute error
    print("\nFeatures correlated with prediction error:")
    print("(Positive = larger errors when feature is higher)")

    error_df = df.with_columns([
        pl.Series('abs_error', abs_errors)
    ])

    error_correlations = calculate_correlations(error_df, all_features, 'abs_error')

    print(f"\n{'Rank':<6} {'Feature':<35} {'r with |error|':<15}")
    print("-"*80)
    for rank, (feature, corr, n) in enumerate(error_correlations, 1):
        marker = " **" if abs(corr) > 0.1 else ""
        print(f"{rank:<6} {feature:<35} {corr:>+.4f}{marker}")

    # Summary
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    print("\n1. MOST PREDICTIVE FEATURES (top 5):")
    for rank, (feature, corr, n) in enumerate(correlations[:5], 1):
        print(f"   {rank}. {feature:<35} r={corr:+.4f}")

    print("\n2. LEAST PREDICTIVE FEATURES (bottom 5):")
    for rank, (feature, corr, n) in enumerate(correlations[-5:], 1):
        print(f"   {rank}. {feature:<35} r={corr:+.4f}")

    print("\n3. ROUTE PARTICIPATION FEATURES:")
    route_feats = [(f, c) for f, c, n in correlations if 'route' in f.lower()]
    for feat, corr in route_feats:
        rank = [i for i, (f, _, _) in enumerate(correlations, 1) if f == feat][0]
        print(f"   Rank {rank:2}: {feat:<35} r={corr:+.4f}")

    if high_corr_pairs:
        print(f"\n4. REDUNDANT FEATURES (|r| > 0.7): {len(high_corr_pairs)} pairs")
        for feat1, feat2, corr in high_corr_pairs[:3]:
            print(f"   {feat1} <-> {feat2} (r={corr:+.4f})")

    print("\n5. FEATURES ASSOCIATED WITH ERRORS:")
    error_culprits = [(f, c) for f, c, n in error_correlations if abs(c) > 0.1]
    if error_culprits:
        for feat, corr in error_culprits[:5]:
            print(f"   {feat:<35} r={corr:+.4f}")
    else:
        print("   No features strongly associated with errors (|r| < 0.1)")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
