"""
Analyze Feature Correlations: Actual Yards vs Beating Betting Lines

Compares which features correlate with:
1. Actual receiving yards (continuous target)
2. Beating the betting line (binary target)

This reveals which features help with betting vs just predicting yards.
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
    """Load the 29-feature dataset from Nov 8 (before optimization)."""
    dataset_path = Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2009_2024_high_volume.parquet"

    if not dataset_path.exists():
        raise FileNotFoundError(f"29-feature dataset not found: {dataset_path}")

    df = pl.read_parquet(dataset_path)
    print(f"Loaded 29-feature dataset: {len(df):,} examples, {len(df.columns)} columns")
    return df


def load_2024_player_prop_lines():
    """Load 2024 player prop betting lines from cache."""
    props_dir = Path(CACHE_DIR) / "player_props" / "2024"
    lines_dict = {}

    if not props_dir.exists():
        print(f"WARNING: Props directory not found: {props_dir}")
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
            except Exception as e:
                continue

    print(f"Loaded {len(lines_dict):,} player prop lines for 2024")
    return lines_dict


def match_dataset_with_lines(df, lines_dict):
    """Match dataset with betting lines and create beat_line target."""
    matched_data = []

    # Filter to 2024 only
    df_2024 = df.filter(pl.col('year') == 2024)
    print(f"Filtered to 2024: {len(df_2024):,} examples")

    for idx in range(len(df_2024)):
        row = df_2024[idx]
        player_id = row['player_id'][0] if hasattr(row['player_id'], '__getitem__') else row['player_id']
        week = row['week'][0] if hasattr(row['week'], '__getitem__') else row['week']
        actual_yards = row['target'][0] if hasattr(row['target'], '__getitem__') else row['target']

        key = (player_id, week)
        if key in lines_dict:
            betting_line = lines_dict[key]
            beat_line = 1 if actual_yards > betting_line else 0

            # Extract all feature values
            feature_dict = {}
            for col in df_2024.columns:
                if col not in ['player_id', 'year', 'week', 'target']:
                    val = row[col][0] if hasattr(row[col], '__getitem__') else row[col]
                    feature_dict[col] = val

            matched_data.append({
                'player_id': player_id,
                'week': week,
                'actual_yards': actual_yards,
                'betting_line': betting_line,
                'beat_line': beat_line,
                **feature_dict
            })

    matched_df = pl.DataFrame(matched_data)
    print(f"Matched {len(matched_df):,} examples with betting lines")
    return matched_df


def calculate_correlations(df, features, target_col):
    """Calculate Pearson or point-biserial correlation between features and target."""
    correlations = []

    for feature in features:
        if feature not in df.columns:
            continue

        # Filter to non-null values for both feature and target
        valid_mask = df[feature].is_not_null() & df[target_col].is_not_null()
        feature_vals = df.filter(valid_mask)[feature].to_numpy()
        target_vals = df.filter(valid_mask)[target_col].to_numpy()

        if len(feature_vals) < 10:
            continue

        # Skip non-numeric features (categorical strings)
        if not np.issubdtype(feature_vals.dtype, np.number):
            continue

        # Skip if no variance
        try:
            if np.std(feature_vals) == 0 or np.std(target_vals) == 0:
                continue
        except (TypeError, ValueError):
            # Skip if can't calculate std (non-numeric)
            continue

        # Calculate correlation
        try:
            corr, p_value = stats.pearsonr(feature_vals, target_vals)
        except (ValueError, TypeError):
            # Skip if correlation calculation fails
            continue

        correlations.append({
            'feature': feature,
            'correlation': corr,
            'p_value': p_value,
            'n_samples': len(feature_vals)
        })

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return correlations


def main():
    print("="*80)
    print("FEATURE CORRELATION ANALYSIS: YARDS vs BEATING THE LINE")
    print("="*80)

    # Step 1: Load 29-feature dataset
    print("\nStep 1: Loading 29-feature dataset...")
    df = load_29_feature_dataset()

    # Identify feature columns (exclude metadata)
    metadata_cols = {'player_id', 'year', 'week', 'target'}
    all_cols = set(df.columns)
    feature_cols = sorted(all_cols - metadata_cols)
    print(f"Identified {len(feature_cols)} feature columns")

    # Step 2: Load betting lines
    print("\nStep 2: Loading 2024 player prop betting lines...")
    lines_dict = load_2024_player_prop_lines()

    if len(lines_dict) == 0:
        print("ERROR: No betting lines found. Cannot proceed.")
        return

    # Step 3: Match dataset with lines
    print("\nStep 3: Matching dataset with betting lines...")
    matched_df = match_dataset_with_lines(df, lines_dict)

    if len(matched_df) == 0:
        print("ERROR: No matches found. Cannot proceed.")
        return

    # Step 4: Run dual correlation analysis
    print("\nStep 4: Running correlation analysis...")
    print("\n" + "-"*80)
    print("ANALYSIS A: Features vs Actual Yards (Continuous Target)")
    print("-"*80)
    corr_yards = calculate_correlations(matched_df, feature_cols, 'actual_yards')

    print(f"\nALL {len(corr_yards)} features correlated with actual yards:")
    print(f"{'Rank':<5} {'Feature':<35} {'Correlation':<12} {'P-value':<10} {'N':<6}")
    print("-"*80)
    for i, item in enumerate(corr_yards, 1):
        print(f"{i:<5} {item['feature']:<35} {item['correlation']:+11.4f} {item['p_value']:>9.4f} {item['n_samples']:>6}")

    print("\n" + "-"*80)
    print("ANALYSIS B: Features vs Beating the Line (Binary Target)")
    print("-"*80)
    corr_beat_line = calculate_correlations(matched_df, feature_cols, 'beat_line')

    print(f"\nALL {len(corr_beat_line)} features correlated with beating the line:")
    print(f"{'Rank':<5} {'Feature':<35} {'Correlation':<12} {'P-value':<10} {'N':<6}")
    print("-"*80)
    for i, item in enumerate(corr_beat_line, 1):
        print(f"{i:<5} {item['feature']:<35} {item['correlation']:+11.4f} {item['p_value']:>9.4f} {item['n_samples']:>6}")

    # Step 5: Generate comparison report
    print("\n" + "="*80)
    print("COMPARISON REPORT: Yards vs Beat Line Correlations")
    print("="*80)

    # Create lookup dicts
    yards_lookup = {item['feature']: item['correlation'] for item in corr_yards}
    beat_line_lookup = {item['feature']: item['correlation'] for item in corr_beat_line}

    # Combine and calculate differences
    comparison = []
    for feature in feature_cols:
        r_yards = yards_lookup.get(feature, 0.0)
        r_beat_line = beat_line_lookup.get(feature, 0.0)

        # Handle NaN values
        if np.isnan(r_yards):
            r_yards = 0.0
        if np.isnan(r_beat_line):
            r_beat_line = 0.0

        diff = r_beat_line - r_yards

        comparison.append({
            'feature': feature,
            'r_yards': r_yards,
            'r_beat_line': r_beat_line,
            'diff': diff,
            'abs_diff': abs(diff)
        })

    # Sort by absolute difference (biggest discrepancies)
    comparison.sort(key=lambda x: x['abs_diff'], reverse=True)

    print("\nALL FEATURES - Complete Comparison:")
    print(f"{'Rank':<5} {'Feature':<35} {'r(yards)':<12} {'r(beat_line)':<12} {'Difference':<12}")
    print("-"*85)
    for i, item in enumerate(comparison, 1):
        print(f"{i:<5} {item['feature']:<35} {item['r_yards']:+11.4f} {item['r_beat_line']:+11.4f} {item['diff']:+11.4f}")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Features good for yards but bad for beating line
    print("\nFeatures that predict YARDS but DON'T help BEAT THE LINE:")
    print("(Sportsbooks likely already price these in)")
    overpriced = [x for x in comparison if abs(x['r_yards']) > 0.1 and x['diff'] < -0.05]
    for item in overpriced[:5]:
        print(f"  - {item['feature']:<35} r_yards={item['r_yards']:+.3f}, r_beat_line={item['r_beat_line']:+.3f}")

    # Features bad for yards but good for beating line
    print("\nFeatures that DON'T predict YARDS but DO help BEAT THE LINE:")
    print("(Potential market inefficiencies)")
    underpriced = [x for x in comparison if abs(x['r_yards']) < 0.05 and abs(x['r_beat_line']) > 0.05]
    for item in underpriced[:5]:
        print(f"  - {item['feature']:<35} r_yards={item['r_yards']:+.3f}, r_beat_line={item['r_beat_line']:+.3f}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total features analyzed: {len(feature_cols)}")
    print(f"Matched examples (2024 with betting lines): {len(matched_df):,}")
    print(f"Beat line rate: {matched_df['beat_line'].mean():.1%} (ideally should be ~50%)")
    print(f"Average actual yards: {matched_df['actual_yards'].mean():.1f}")
    print(f"Average betting line: {matched_df['betting_line'].mean():.1f}")

    # Correlation of correlations (filter out NaNs)
    r_yards_list = [yards_lookup.get(f, 0) for f in feature_cols]
    r_beat_line_list = [beat_line_lookup.get(f, 0) for f in feature_cols]

    # Filter out NaN values
    valid_pairs = [(y, b) for y, b in zip(r_yards_list, r_beat_line_list) if not (np.isnan(y) or np.isnan(b))]
    if len(valid_pairs) > 2:
        valid_y, valid_b = zip(*valid_pairs)
        meta_corr, _ = stats.pearsonr(valid_y, valid_b)
        print(f"\nCorrelation between r_yards and r_beat_line: {meta_corr:.3f}")
        print(f"(How similar are the two signals? 1.0 = identical, 0.0 = unrelated)")
        print(f"Based on {len(valid_pairs)} features with valid correlations")
    else:
        print("\nNot enough valid features to calculate meta-correlation")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
