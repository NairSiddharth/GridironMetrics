"""
Generic Prop Type Evaluation with Range Analysis

Evaluates a trained ML ensemble on 2023 & 2024 test data with comprehensive
range analysis for optimal threshold selection.

Usage:
    python scripts/evaluate_prop_type.py passing_yards
    python scripts/evaluate_prop_type.py passing_tds
    python scripts/evaluate_prop_type.py rushing_yards
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from modules.ml_ensemble import PropEnsembleModel
from modules.ml_training_data_builder import TrainingDataBuilder
from modules.constants import CACHE_DIR
from modules.prop_types import get_display_name
from tests.backtest_props import load_historical_props_for_week
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


def load_real_betting_lines_for_year(year: int, prop_type: str) -> dict:
    """
    Load all real betting lines for a specific year and prop type.

    Args:
        year: Season year
        prop_type: Prop type (e.g., 'passing_yards')

    Returns:
        Dict mapping (gsis_id, week) -> betting_line
    """
    prop_display_name = get_display_name(prop_type)
    lines_lookup = {}

    # Load props for each week (weeks 1-18 for 2024, fewer for other years)
    max_week = 18 if year == 2024 else 17

    for week in range(1, max_week + 1):
        # Use tuesday lines (opening lines, most stable)
        historical_props = load_historical_props_for_week(year, week, day='tuesday')

        if not historical_props:
            continue

        for player_data in historical_props:
            gsis_id = player_data.get('gsis_id')
            props = player_data.get('props', {})

            if not gsis_id or prop_display_name not in props:
                continue

            betting_line = props[prop_display_name]['line']
            lines_lookup[(gsis_id, week)] = betting_line

    return lines_lookup


def evaluate_prop_type(prop_type: str):
    """
    Evaluate ML model for a specific prop type on held-out test years.

    Args:
        prop_type: Prop type to evaluate (e.g., 'passing_yards', 'passing_tds')
    """

    print("\n" + "="*80)
    print(f"ML MODEL MULTI-YEAR EVALUATION - {prop_type.upper()}")
    print("="*80)
    print("\nTraining: 2015-2022")
    print("Testing: 2023 & 2024")

    # Load full training data
    training_data_path = Path(CACHE_DIR) / "ml_training_data" / f"{prop_type}_2015_2024.parquet"

    if not training_data_path.exists():
        print(f"\nWARNING: Training data not found at {training_data_path}")
        print(f"Generating training data for {prop_type}...")

        # Generate data
        builder = TrainingDataBuilder()
        full_df = builder.build_training_dataset(
            start_year=2015,
            end_year=2024,
            prop_type=prop_type
        )

        if len(full_df) == 0:
            print(f"ERROR: Failed to generate training data for {prop_type}")
            return None
    else:
        print(f"\nLoading data...")
        full_df = pl.read_parquet(training_data_path)

    # Check years available
    years_available = full_df['year'].unique().sort().to_list()
    print(f"Years available: {years_available}")

    # Split data
    train_df = full_df.filter(pl.col('year') < 2023)
    test_2023 = full_df.filter(pl.col('year') == 2023)
    test_2024 = full_df.filter(pl.col('year') == 2024)

    print(f"\nDataset Split (before matching real lines):")
    print(f"  Train (2015-2022): {len(train_df):,} examples")
    print(f"  Test 2023:         {len(test_2023):,} examples")
    print(f"  Test 2024:         {len(test_2024):,} examples")
    print(f"  Total test:        {len(test_2023) + len(test_2024):,} examples")

    # Load real betting lines for test years
    print(f"\nLoading real betting lines...")
    lines_2023 = load_real_betting_lines_for_year(2023, prop_type)
    lines_2024 = load_real_betting_lines_for_year(2024, prop_type)

    print(f"  2023: {len(lines_2023)} real betting lines")
    print(f"  2024: {len(lines_2024)} real betting lines")

    # Check if we have prop_data in cache/player_props/
    if len(lines_2023) == 0 and len(lines_2024) == 0:
        print(f"\nWARNING: No real betting lines found in cache/player_props/")
        print(f"  This prop type may not be available in our betting data.")
        print(f"  Available data is primarily for 2024 season.")
        print(f"\n  Falling back to evaluation without real lines.")
        return None

    # Match test data to real betting lines
    # Add betting_line column by matching (player_id, week)
    def add_real_lines(df: pl.DataFrame, lines_dict: dict) -> pl.DataFrame:
        """Add real betting lines to dataframe by matching player_id and week."""
        if len(df) == 0:
            return df.with_columns(pl.lit(None).alias('betting_line'))

        # Convert to pandas for easier matching
        pdf = df.to_pandas()
        pdf['betting_line'] = pdf.apply(
            lambda row: lines_dict.get((row['player_id'], row['week']), None),
            axis=1
        )

        # Convert back to polars and filter to only rows with real lines
        matched_df = pl.from_pandas(pdf)
        matched_df = matched_df.filter(pl.col('betting_line').is_not_null())

        return matched_df

    test_2023 = add_real_lines(test_2023, lines_2023)
    test_2024 = add_real_lines(test_2024, lines_2024)

    print(f"\nDataset Split (after matching real lines):")
    print(f"  Test 2023:         {len(test_2023):,} examples with real lines")
    print(f"  Test 2024:         {len(test_2024):,} examples with real lines")
    print(f"  Total test:        {len(test_2023) + len(test_2024):,} examples")

    if len(test_2023) + len(test_2024) == 0:
        print(f"\nERROR: No test examples matched to real betting lines")
        print(f"  Cannot evaluate without real market data.")
        return None

    # Train model on 2015-2022
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL (2015-2022)")
    print(f"{'='*80}")

    ensemble = PropEnsembleModel(prop_type=prop_type)
    ensemble.train(train_df, n_splits=3, verbose=False)

    print(f"\nModel trained. Generating predictions...")

    # Evaluate on 2023
    print(f"\n{'='*80}")
    print(f"TESTING ON 2023")
    print(f"{'='*80}")

    if len(test_2023) == 0:
        print(f"\nWARNING: No 2023 test examples with real betting lines - skipping 2023 evaluation")
        predictions_2023 = np.array([])
        actuals_2023 = np.array([])
        real_lines_2023 = np.array([])
    else:
        predictions_2023 = ensemble.predict_batch(test_2023)
        actuals_2023 = test_2023['target'].to_numpy()
        real_lines_2023 = test_2023['betting_line'].to_numpy()

        mae_2023 = np.mean(np.abs(predictions_2023 - actuals_2023))
        rmse_2023 = np.sqrt(np.mean((predictions_2023 - actuals_2023) ** 2))

        print(f"Test examples: {len(test_2023):,} (matched to real lines)")
        print(f"\nPrediction Accuracy:")
        print(f"  MAE:  {mae_2023:.2f}")
        print(f"  RMSE: {rmse_2023:.2f}")

        # Evaluate against REAL betting lines from 2023
        ml_predicts_over_2023 = predictions_2023 > real_lines_2023
        actual_was_over_2023 = actuals_2023 > real_lines_2023
        correct_2023 = ml_predicts_over_2023 == actual_was_over_2023

        accuracy_2023 = correct_2023.sum() / len(correct_2023)
        wins_2023 = correct_2023.sum()
        total_2023 = len(correct_2023)
        roi_2023 = ((wins_2023 * 210 - total_2023 * 110) / (total_2023 * 110)) * 100

        print(f"\nBetting Performance:")
        print(f"  Accuracy: {accuracy_2023*100:.1f}%")
        print(f"  Break-even: 52.4%")
        print(f"  Edge: {(accuracy_2023 - 0.524)*100:+.1f}%")
        print(f"  ROI: {roi_2023:+.2f}%")

    # Evaluate on 2024
    print(f"\n{'='*80}")
    print(f"TESTING ON 2024")
    print(f"{'='*80}")

    if len(test_2024) == 0:
        print(f"\nWARNING: No 2024 test examples with real betting lines - skipping 2024 evaluation")
        predictions_2024 = np.array([])
        actuals_2024 = np.array([])
        real_lines_2024 = np.array([])
    else:
        predictions_2024 = ensemble.predict_batch(test_2024)
        actuals_2024 = test_2024['target'].to_numpy()
        real_lines_2024 = test_2024['betting_line'].to_numpy()

        mae_2024 = np.mean(np.abs(predictions_2024 - actuals_2024))
        rmse_2024 = np.sqrt(np.mean((predictions_2024 - actuals_2024) ** 2))

        print(f"Test examples: {len(test_2024):,} (matched to real lines)")
        print(f"\nPrediction Accuracy:")
        print(f"  MAE:  {mae_2024:.2f}")
        print(f"  RMSE: {rmse_2024:.2f}")

        # Evaluate against REAL betting lines from 2024
        ml_predicts_over_2024 = predictions_2024 > real_lines_2024
        actual_was_over_2024 = actuals_2024 > real_lines_2024
        correct_2024 = ml_predicts_over_2024 == actual_was_over_2024

        accuracy_2024 = correct_2024.sum() / len(correct_2024)
        wins_2024 = correct_2024.sum()
        total_2024 = len(correct_2024)
        roi_2024 = ((wins_2024 * 210 - total_2024 * 110) / (total_2024 * 110)) * 100

        print(f"\nBetting Performance:")
        print(f"  Accuracy: {accuracy_2024*100:.1f}%")
        print(f"  Break-even: 52.4%")
        print(f"  Edge: {(accuracy_2024 - 0.524)*100:+.1f}%")
        print(f"  ROI: {roi_2024:+.2f}%")

    # Combined results
    print(f"\n{'='*80}")
    print(f"COMBINED RESULTS (2023 + 2024)")
    print(f"{'='*80}")

    # Only proceed if we have at least some test data
    if len(test_2023) == 0 and len(test_2024) == 0:
        print(f"\nERROR: No test data available from either year")
        print(f"  Cannot perform combined evaluation.")
        return None

    # Combine data from years that have data
    test_dfs = []
    all_predictions = []
    all_actuals = []
    all_lines = []

    if len(test_2023) > 0:
        test_dfs.append(test_2023)
        all_predictions.append(predictions_2023)
        all_actuals.append(actuals_2023)
        all_lines.append(real_lines_2023)

    if len(test_2024) > 0:
        test_dfs.append(test_2024)
        all_predictions.append(predictions_2024)
        all_actuals.append(actuals_2024)
        all_lines.append(real_lines_2024)

    combined_test = pl.concat(test_dfs)
    combined_predictions = np.concatenate(all_predictions)
    combined_actuals = np.concatenate(all_actuals)
    combined_real_lines = np.concatenate(all_lines)

    combined_mae = np.mean(np.abs(combined_predictions - combined_actuals))
    combined_rmse = np.sqrt(np.mean((combined_predictions - combined_actuals) ** 2))

    # Evaluate against combined REAL betting lines
    combined_over = combined_predictions > combined_real_lines
    combined_actual_over = combined_actuals > combined_real_lines
    combined_correct = combined_over == combined_actual_over
    combined_accuracy = combined_correct.sum() / len(combined_correct)

    combined_wins = combined_correct.sum()
    combined_total = len(combined_correct)
    combined_roi = ((combined_wins * 210 - combined_total * 110) / (combined_total * 110)) * 100

    print(f"\nTotal Test Examples: {len(combined_test):,}")
    print(f"\nPrediction Accuracy:")
    print(f"  MAE:  {combined_mae:.2f}")
    print(f"  RMSE: {combined_rmse:.2f}")

    print(f"\nBetting Performance:")
    print(f"  Accuracy: {combined_accuracy*100:.1f}%")
    print(f"  Wins: {combined_wins} / {combined_total}")
    print(f"  ROI: {combined_roi:+.2f}%")

    # Confidence analysis on combined data (using REAL betting lines)
    combined_confidence = np.abs(combined_predictions - combined_real_lines)

    # Find optimal threshold with range analysis
    print(f"\n{'='*80}")
    print(f"SELECTIVE BETTING ANALYSIS")
    print(f"{'='*80}")

    # Determine appropriate threshold range based on prop type
    if 'yards' in prop_type:
        threshold_range = range(20, 125, 5)  # 20-120 yards
        unit = "yards"
    elif 'tds' in prop_type:
        # For TDs, use smaller increments (0.2 TDs = 0.2)
        threshold_range = [i * 0.1 for i in range(2, 31)]  # 0.2-3.0 TDs
        unit = "TDs"
    elif 'receptions' in prop_type:
        threshold_range = [i * 0.5 for i in range(1, 21)]  # 0.5-10 receptions
        unit = "rec"
    else:
        # Default to yards
        threshold_range = range(10, 100, 5)
        unit = "units"

    print(f"\nTesting different confidence thresholds:")
    print(f"\n{'Threshold':<15} {'Accuracy':<12} {'# Bets':<12} {'ROI':<12} {'Status':<12}")
    print("-" * 75)

    # Store detailed results for range analysis
    threshold_results = []

    for threshold in threshold_range:
        mask = combined_confidence >= threshold
        if mask.sum() >= 10:  # Minimum 10 bets (lowered from 20 for better visibility)
            filtered_accuracy = combined_correct[mask].sum() / mask.sum()
            filtered_wins = combined_correct[mask].sum()
            filtered_total = mask.sum()
            filtered_roi = ((filtered_wins * 210 - filtered_total * 110) / (filtered_total * 110)) * 100

            status = "PROFITABLE" if filtered_accuracy > 0.524 else "Unprofitable"

            threshold_results.append({
                'threshold': threshold,
                'accuracy': filtered_accuracy,
                'roi': filtered_roi,
                'bets': filtered_total,
                'profitable': filtered_accuracy > 0.524
            })

            # Print ALL thresholds for yards (show full picture)
            if 'yards' in prop_type:
                print(f"{threshold:>6.1f} {unit:<6}  {filtered_accuracy*100:>6.1f}%      {filtered_total:>6}       {filtered_roi:>+6.2f}%      {status}")
            # For TDs and receptions, print every 0.5 increment
            elif ('tds' in prop_type and abs(threshold - round(threshold * 2) / 2) < 0.01) or \
                 ('receptions' in prop_type and threshold % 0.5 == 0):
                print(f"{threshold:>6.1f} {unit:<6}  {filtered_accuracy*100:>6.1f}%      {filtered_total:>6}       {filtered_roi:>+6.2f}%      {status}")

    # Identify profitable ranges
    print(f"\n{'='*80}")
    print(f"OPTIMAL THRESHOLD RANGES")
    print(f"{'='*80}")

    if threshold_results:
        # Find consecutive profitable thresholds
        ranges = []
        current_range = []

        for i, result in enumerate(threshold_results):
            if result['profitable']:
                current_range.append(result)
            else:
                if current_range:
                    ranges.append(current_range)
                    current_range = []

        # Don't forget the last range
        if current_range:
            ranges.append(current_range)

        if ranges:
            print(f"\nFound {len(ranges)} profitable range(s):\n")

            for idx, range_data in enumerate(ranges, 1):
                min_threshold = range_data[0]['threshold']
                max_threshold = range_data[-1]['threshold']
                avg_accuracy = np.mean([r['accuracy'] for r in range_data])
                avg_roi = np.mean([r['roi'] for r in range_data])
                total_bets_range = range_data[0]['bets']  # At lowest threshold

                print(f"Range {idx}: {min_threshold:.1f}-{max_threshold:.1f} {unit}")
                print(f"  Avg Accuracy: {avg_accuracy*100:.1f}%")
                print(f"  Avg ROI: {avg_roi:+.2f}%")
                print(f"  Bets (at {min_threshold:.1f} {unit}): {total_bets_range}")

                # Show best performing threshold in this range
                best_in_range = max(range_data, key=lambda x: x['roi'])
                print(f"  Peak: {best_in_range['threshold']:.1f} {unit} ({best_in_range['accuracy']*100:.1f}% acc, {best_in_range['roi']:+.2f}% ROI, {best_in_range['bets']} bets)")
                print()

            # Strategy recommendations
            print(f"{'='*80}")
            print(f"STRATEGY RECOMMENDATIONS")
            print(f"{'='*80}")

            # Find the main profitable range (usually the largest)
            main_range = max(ranges, key=lambda r: len(r))

            # Conservative: higher threshold, lower volume, higher accuracy
            conservative_data = main_range[-1] if len(main_range) > 1 else main_range[0]

            # Aggressive: lower threshold, higher volume, still profitable
            aggressive_data = main_range[0]

            # Balanced: middle of the range
            mid_idx = len(main_range) // 2
            balanced_data = main_range[mid_idx]

            print(f"\n1. CONSERVATIVE (High Accuracy, Low Volume)")
            print(f"   Threshold: {conservative_data['threshold']:.1f}+ {unit}")
            print(f"   Expected: {conservative_data['accuracy']*100:.1f}% accuracy, {conservative_data['roi']:+.2f}% ROI")
            print(f"   Volume: ~{conservative_data['bets']} bets/2 years (~{conservative_data['bets']/34:.1f} bets/week)")

            print(f"\n2. BALANCED (Good Accuracy, Moderate Volume)")
            print(f"   Threshold: {balanced_data['threshold']:.1f}+ {unit}")
            print(f"   Expected: {balanced_data['accuracy']*100:.1f}% accuracy, {balanced_data['roi']:+.2f}% ROI")
            print(f"   Volume: ~{balanced_data['bets']} bets/2 years (~{balanced_data['bets']/34:.1f} bets/week)")

            print(f"\n3. AGGRESSIVE (More Bets, Still Profitable)")
            print(f"   Threshold: {aggressive_data['threshold']:.1f}+ {unit}")
            print(f"   Expected: {aggressive_data['accuracy']*100:.1f}% accuracy, {aggressive_data['roi']:+.2f}% ROI")
            print(f"   Volume: ~{aggressive_data['bets']} bets/2 years (~{aggressive_data['bets']/34:.1f} bets/week)")

            print(f"\nRECOMMENDATION: Start with BALANCED strategy ({balanced_data['threshold']:.1f}+ {unit})")
            print(f"   This gives you ~{balanced_data['bets']/34:.1f} bets/week with strong profitability.")
            print(f"   Once validated in live betting, consider AGGRESSIVE for more volume.")
        else:
            print(f"\nERROR: No profitable threshold ranges found.")
            print(f"   Model needs improvement before live betting.")
    else:
        print(f"\nERROR: Insufficient data for range analysis.")

    print(f"\n{'='*80}")

    return {
        'prop_type': prop_type,
        'combined_mae': combined_mae,
        'combined_accuracy': combined_accuracy,
        'combined_roi': combined_roi,
        'threshold_results': threshold_results,
        'ranges': ranges if threshold_results and ranges else []
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate_prop_type.py <prop_type>")
        print("\nAvailable prop types:")
        print("  - passing_yards")
        print("  - passing_tds")
        print("  - rushing_yards")
        print("  - rushing_tds")
        print("  - receptions")
        print("  - receiving_yards")
        print("  - receiving_tds")
        sys.exit(1)

    prop_type = sys.argv[1]
    results = evaluate_prop_type(prop_type)
