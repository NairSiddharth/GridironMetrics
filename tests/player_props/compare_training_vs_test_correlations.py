"""
Compare Feature Correlations: Training Set vs Test Set

Analyzes whether feature correlations are stable between:
1. Training set (2016-2023, ~7k examples) - what model learns from
2. Test set (2024, ~288 examples with betting lines) - what model is evaluated on

This reveals if correlation-guided feature selection failed due to distribution shift.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import numpy as np
import json
from scipy import stats
from modules.constants import CACHE_DIR


def load_29_feature_dataset():
    """Load the 29-feature dataset."""
    dataset_path = Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2009_2024_high_volume.parquet"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pl.read_parquet(dataset_path)
    print(f"Loaded dataset: {len(df):,} examples, {len(df.columns)} columns")
    return df


def load_2024_player_prop_lines():
    """Load 2024 player prop betting lines."""
    props_dir = Path(CACHE_DIR) / "player_props" / "2024"
    lines_dict = {}

    if not props_dir.exists():
        return lines_dict

    for player_dir in props_dir.iterdir():
        if not player_dir.is_dir():
            continue

        for week_dir in player_dir.iterdir():
            if not week_dir.is_dir() or not week_dir.name.startswith('week'):
                continue

            friday_file = week_dir / "friday.json"
            if not friday_file.exists():
                continue

            try:
                with open(friday_file, 'r') as f:
                    data = json.load(f)

                if data.get('position') != 'WR':
                    continue

                recv_yards_data = data.get('props', {}).get('Receiving Yards')
                if recv_yards_data and 'line' in recv_yards_data:
                    gsis_id = data['gsis_id']
                    week_num = data['metadata']['week']
                    key = (gsis_id, week_num)
                    lines_dict[key] = recv_yards_data['line']
            except:
                continue

    print(f"Loaded {len(lines_dict):,} betting lines for 2024")
    return lines_dict


def calculate_correlations(df, features, target_col, dataset_name):
    """Calculate Pearson correlation between features and target."""
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS: {dataset_name}")
    print(f"{'='*80}")
    print(f"Sample size: {len(df):,} examples")
    print(f"Target: {target_col}")

    correlations = []

    for feature in features:
        if feature not in df.columns:
            continue

        # Filter to non-null values
        valid_mask = df[feature].is_not_null() & df[target_col].is_not_null()
        feature_vals = df.filter(valid_mask)[feature].to_numpy()
        target_vals = df.filter(valid_mask)[target_col].to_numpy()

        if len(feature_vals) < 10:
            continue

        # Skip non-numeric
        if not np.issubdtype(feature_vals.dtype, np.number):
            continue

        # Skip if no variance
        try:
            if np.std(feature_vals) == 0 or np.std(target_vals) == 0:
                continue
        except:
            continue

        # Calculate correlation
        try:
            corr, p_value = stats.pearsonr(feature_vals, target_vals)
        except:
            continue

        correlations.append({
            'feature': feature,
            'correlation': corr,
            'p_value': p_value,
            'n_samples': len(feature_vals)
        })

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

    print(f"\nALL {len(correlations)} features by absolute correlation:")
    print(f"{'Rank':<5} {'Feature':<35} {'Correlation':<12} {'P-value':<10} {'N':<6}")
    print("-"*80)
    for i, item in enumerate(correlations, 1):
        corr_str = f"{item['correlation']:+11.4f}" if not np.isnan(item['correlation']) else "       NaN"
        pval_str = f"{item['p_value']:>9.4f}" if not np.isnan(item['p_value']) else "      nan"
        print(f"{i:<5} {item['feature']:<35} {corr_str} {pval_str} {item['n_samples']:>6}")

    return correlations


def main():
    print("="*80)
    print("TRAINING SET vs TEST SET CORRELATION COMPARISON")
    print("="*80)

    # Load full dataset
    print("\nStep 1: Loading full dataset...")
    df = load_29_feature_dataset()

    # Identify features
    metadata_cols = {'player_id', 'year', 'week', 'target'}
    feature_cols = sorted(set(df.columns) - metadata_cols)
    print(f"Identified {len(feature_cols)} feature columns")

    # Filter to training set (2016-2023)
    df_train = df.filter((pl.col('year') >= 2016) & (pl.col('year') <= 2023))
    print(f"\nTraining set (2016-2023): {len(df_train):,} examples")

    # Filter to test set (2024)
    df_test = df.filter(pl.col('year') == 2024)
    print(f"Test set (2024): {len(df_test):,} examples")

    # Analysis 1: Training set correlations (yards)
    print("\n" + "="*80)
    print("ANALYSIS 1: Training Set (2016-2023) - Feature -> Yards")
    print("="*80)
    corr_train = calculate_correlations(df_train, feature_cols, 'target',
                                        "Training Set (2016-2023, N=7,012)")

    # Analysis 2: Test set correlations (yards)
    print("\n" + "="*80)
    print("ANALYSIS 2: Test Set (2024) - Feature -> Yards")
    print("="*80)
    corr_test = calculate_correlations(df_test, feature_cols, 'target',
                                      "Test Set (2024, N=1,012)")

    # Analysis 3: Test set with betting lines
    print("\n" + "="*80)
    print("ANALYSIS 3: Test Set with Betting Lines (2024) - Feature -> Yards & Beat Line")
    print("="*80)

    lines_dict = load_2024_player_prop_lines()

    # Match test set with betting lines
    matched_data = []
    for idx in range(len(df_test)):
        row = df_test[idx]
        player_id = row['player_id'][0] if hasattr(row['player_id'], '__getitem__') else row['player_id']
        week = row['week'][0] if hasattr(row['week'], '__getitem__') else row['week']
        actual_yards = row['target'][0] if hasattr(row['target'], '__getitem__') else row['target']

        key = (player_id, week)
        if key in lines_dict:
            betting_line = lines_dict[key]
            beat_line = 1 if actual_yards > betting_line else 0

            feature_dict = {}
            for col in df_test.columns:
                if col not in ['player_id', 'year', 'week', 'target']:
                    val = row[col][0] if hasattr(row[col], '__getitem__') else row[col]
                    feature_dict[col] = val

            matched_data.append({
                'actual_yards': actual_yards,
                'betting_line': betting_line,
                'beat_line': beat_line,
                **feature_dict
            })

    df_matched = pl.DataFrame(matched_data)
    print(f"Matched {len(df_matched):,} examples with betting lines")

    corr_matched_yards = calculate_correlations(df_matched, feature_cols, 'actual_yards',
                                                "Test Set with Betting Lines (2024, N=288) - Yards")

    corr_matched_beat = calculate_correlations(df_matched, feature_cols, 'beat_line',
                                              "Test Set with Betting Lines (2024, N=288) - Beat Line")

    # Comparison analysis
    print("\n" + "="*80)
    print("STABILITY ANALYSIS: How much do correlations change?")
    print("="*80)

    # Create lookups
    train_lookup = {item['feature']: item['correlation'] for item in corr_train}
    test_lookup = {item['feature']: item['correlation'] for item in corr_test}
    matched_lookup = {item['feature']: item['correlation'] for item in corr_matched_yards}
    beat_lookup = {item['feature']: item['correlation'] for item in corr_matched_beat}

    # Calculate stability metrics
    comparison = []
    for feature in feature_cols:
        r_train = train_lookup.get(feature, 0.0)
        r_test = test_lookup.get(feature, 0.0)
        r_matched = matched_lookup.get(feature, 0.0)
        r_beat = beat_lookup.get(feature, 0.0)

        # Handle NaN
        if np.isnan(r_train):
            r_train = 0.0
        if np.isnan(r_test):
            r_test = 0.0
        if np.isnan(r_matched):
            r_matched = 0.0
        if np.isnan(r_beat):
            r_beat = 0.0

        diff_train_test = r_test - r_train
        diff_train_matched = r_matched - r_train

        comparison.append({
            'feature': feature,
            'r_train': r_train,
            'r_test': r_test,
            'r_matched': r_matched,
            'r_beat': r_beat,
            'diff_train_test': diff_train_test,
            'diff_train_matched': diff_train_matched,
            'abs_diff': abs(diff_train_matched)
        })

    # Sort by biggest change from training to matched
    comparison.sort(key=lambda x: x['abs_diff'], reverse=True)

    print("\nFeatures with BIGGEST correlation changes (Train -> Matched):")
    print("(Large changes suggest distribution shift between training and test)")
    print(f"{'Rank':<5} {'Feature':<30} {'Train':<10} {'Test':<10} {'Matched':<10} {'Change':<10}")
    print("-"*85)
    for i, item in enumerate(comparison[:20], 1):
        print(f"{i:<5} {item['feature']:<30} {item['r_train']:+9.4f} {item['r_test']:+9.4f} {item['r_matched']:+9.4f} {item['diff_train_matched']:+9.4f}")

    # Correlation between training and test correlations
    r_train_list = [train_lookup.get(f, 0) for f in feature_cols]
    r_test_list = [test_lookup.get(f, 0) for f in feature_cols]
    r_matched_list = [matched_lookup.get(f, 0) for f in feature_cols]

    # Filter NaNs
    valid_pairs_train_test = [(t, s) for t, s in zip(r_train_list, r_test_list) if not (np.isnan(t) or np.isnan(s))]
    valid_pairs_train_matched = [(t, m) for t, m in zip(r_train_list, r_matched_list) if not (np.isnan(t) or np.isnan(m))]

    if len(valid_pairs_train_test) > 2:
        valid_train, valid_test = zip(*valid_pairs_train_test)
        meta_corr_train_test, _ = stats.pearsonr(valid_train, valid_test)
        print(f"\nCorrelation stability (Train vs Test, full 1012): r={meta_corr_train_test:.3f}")
        print(f"(1.0 = perfect stability, 0.0 = completely different)")

    if len(valid_pairs_train_matched) > 2:
        valid_train, valid_matched = zip(*valid_pairs_train_matched)
        meta_corr_train_matched, _ = stats.pearsonr(valid_train, valid_matched)
        print(f"Correlation stability (Train vs Matched, 288): r={meta_corr_train_matched:.3f}")
        print(f"(This is what we used for feature selection!)")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Training set: {len(df_train):,} examples (2016-2023)")
    print(f"Test set: {len(df_test):,} examples (2024)")
    print(f"Matched with betting lines: {len(df_matched):,} examples (2024)")
    print(f"\nStability finding:")
    if len(valid_pairs_train_matched) > 2:
        if meta_corr_train_matched < 0.5:
            print(f"  UNSTABLE: Correlation patterns changed significantly (r={meta_corr_train_matched:.3f})")
            print(f"  This explains why correlation-guided feature reduction failed!")
        elif meta_corr_train_matched < 0.8:
            print(f"  MODERATE: Some correlation shift (r={meta_corr_train_matched:.3f})")
            print(f"  Feature selection based on 288 examples is risky")
        else:
            print(f"  STABLE: Correlations are consistent (r={meta_corr_train_matched:.3f})")
            print(f"  The problem lies elsewhere")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
