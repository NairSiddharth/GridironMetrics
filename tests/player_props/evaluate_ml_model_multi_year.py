"""
ML Model Evaluation - Multi-Year Test (2023 & 2024)

Evaluates the trained ML ensemble on both 2023 and 2024 held-out test data
to validate consistency and robustness of the model.

Training: 2015-2022
Testing: 2023, 2024 (separately and combined)

Usage:
    python scripts/evaluate_ml_model_multi_year.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from modules.ml_ensemble import PropEnsembleModel
from modules.constants import CACHE_DIR
from modules.logger import get_logger

logger = get_logger(__name__)


def evaluate_year(ensemble, test_df, year, verbose=True):
    """
    Evaluate model on a specific test year.

    Args:
        ensemble: Trained ensemble model
        test_df: Test dataframe for the year
        year: Year being tested
        verbose: Print detailed output

    Returns:
        Dictionary of results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"TESTING ON {year}")
        print(f"{'='*80}")
        print(f"Test examples: {len(test_df):,}")

    # Generate predictions
    predictions = ensemble.predict_batch(test_df)
    actuals = test_df['target'].to_numpy()

    # Calculate basic metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    if verbose:
        print(f"\nPrediction Accuracy:")
        print(f"  MAE:  {mae:.2f} yards")
        print(f"  RMSE: {rmse:.2f} yards")

    # Simulate betting lines
    np.random.seed(42 + year)  # Different seed per year
    simulated_lines = actuals + np.random.uniform(-10, 10, size=len(actuals))

    # Binary predictions
    ml_predicts_over = predictions > simulated_lines
    actual_was_over = actuals > simulated_lines
    correct_predictions = ml_predicts_over == actual_was_over

    accuracy = correct_predictions.sum() / len(correct_predictions)

    # Confidence levels
    confidence = np.abs(predictions - simulated_lines)
    q1, q2, q3 = np.percentile(confidence, [25, 50, 75])

    level_stats = []
    for level, (low, high) in enumerate([
        (0, q1),
        (q1, q2),
        (q2, q3),
        (q3, np.inf)
    ], 1):
        mask = (confidence >= low) & (confidence < high)
        if mask.sum() > 0:
            level_accuracy = correct_predictions[mask].sum() / mask.sum()
            avg_confidence = confidence[mask].mean()
            level_stats.append({
                'level': level,
                'accuracy': level_accuracy,
                'avg_confidence': avg_confidence,
                'count': mask.sum()
            })

    # ROI calculation
    wins = correct_predictions.sum()
    losses = len(correct_predictions) - wins
    total_wagered = len(correct_predictions) * 110
    gross_winnings = wins * 210  # Get stake back + profit
    net_profit = gross_winnings - total_wagered
    roi = (net_profit / total_wagered) * 100

    if verbose:
        print(f"\nBetting Performance:")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Break-even: 52.4%")
        print(f"  Edge: {(accuracy - 0.524)*100:+.1f}%")
        print(f"  ROI: {roi:+.2f}%")

        print(f"\nConfidence Levels:")
        for stat in level_stats:
            print(f"  Level {stat['level']} (edge: {stat['avg_confidence']:.1f} yards): "
                  f"{stat['accuracy']*100:.1f}% ({stat['count']} bets)")

    return {
        'year': year,
        'test_examples': len(test_df),
        'mae': mae,
        'rmse': rmse,
        'accuracy': accuracy,
        'roi': roi,
        'wins': wins,
        'losses': losses,
        'level_stats': level_stats,
        'predictions': predictions,
        'actuals': actuals,
        'confidence': confidence
    }


def main():
    print("\n" + "="*80)
    print("ML MODEL MULTI-YEAR EVALUATION")
    print("="*80)
    print("\nTraining: 2015-2022")
    print("Testing: 2023 & 2024")

    # Load full training data
    training_data_path = Path(CACHE_DIR) / "ml_training_data" / "passing_yards_2015_2023.parquet"

    if not training_data_path.exists():
        print(f"\nERROR: Training data not found at {training_data_path}")
        return

    print(f"\nLoading data...")
    full_df = pl.read_parquet(training_data_path)

    # Check if we have 2024 data
    years_available = full_df['year'].unique().sort().to_list()
    print(f"Years available: {years_available}")

    if 2024 not in years_available:
        print(f"\nWARNING: 2024 data not found in training dataset")
        print(f"Regenerating training data to include 2024...")

        # Generate 2024 data
        from modules.ml_training_data_builder import TrainingDataBuilder
        builder = TrainingDataBuilder()

        print(f"\nGenerating 2015-2024 training data...")
        full_df = builder.build_training_dataset(
            start_year=2015,
            end_year=2024,
            prop_type='passing_yards'
        )

        if len(full_df) == 0:
            print("ERROR: Failed to generate training data")
            return

    # Split data
    train_df = full_df.filter(pl.col('year') < 2023)
    test_2023 = full_df.filter(pl.col('year') == 2023)
    test_2024 = full_df.filter(pl.col('year') == 2024)

    print(f"\nDataset Split:")
    print(f"  Train (2015-2022): {len(train_df):,} examples")
    print(f"  Test 2023:         {len(test_2023):,} examples")
    print(f"  Test 2024:         {len(test_2024):,} examples")
    print(f"  Total test:        {len(test_2023) + len(test_2024):,} examples")

    # Train model on 2015-2022
    print(f"\n{'='*80}")
    print("TRAINING MODEL (2015-2022)")
    print(f"{'='*80}")

    ensemble = PropEnsembleModel(prop_type='passing_yards')
    ensemble.train(train_df, n_splits=3, verbose=False)

    print(f"\nModel trained. Generating predictions...")

    # Evaluate on 2023
    results_2023 = evaluate_year(ensemble, test_2023, 2023, verbose=True)

    # Evaluate on 2024
    results_2024 = evaluate_year(ensemble, test_2024, 2024, verbose=True)

    # Combined results
    print(f"\n{'='*80}")
    print("COMBINED RESULTS (2023 + 2024)")
    print(f"{'='*80}")

    combined_test = pl.concat([test_2023, test_2024])
    combined_predictions = np.concatenate([results_2023['predictions'], results_2024['predictions']])
    combined_actuals = np.concatenate([results_2023['actuals'], results_2024['actuals']])

    combined_mae = np.mean(np.abs(combined_predictions - combined_actuals))
    combined_rmse = np.sqrt(np.mean((combined_predictions - combined_actuals) ** 2))

    # Simulate combined betting
    np.random.seed(42)
    combined_lines = combined_actuals + np.random.uniform(-10, 10, size=len(combined_actuals))
    combined_over = combined_predictions > combined_lines
    combined_actual_over = combined_actuals > combined_lines
    combined_correct = combined_over == combined_actual_over
    combined_accuracy = combined_correct.sum() / len(combined_correct)

    combined_wins = combined_correct.sum()
    combined_total = len(combined_correct)
    combined_roi = ((combined_wins * 210 - combined_total * 110) / (combined_total * 110)) * 100

    print(f"\nTotal Test Examples: {len(combined_test):,}")
    print(f"\nPrediction Accuracy:")
    print(f"  MAE:  {combined_mae:.2f} yards")
    print(f"  RMSE: {combined_rmse:.2f} yards")

    print(f"\nBetting Performance:")
    print(f"  Accuracy: {combined_accuracy*100:.1f}%")
    print(f"  Wins: {combined_wins} / {combined_total}")
    print(f"  ROI: {combined_roi:+.2f}%")

    # Confidence analysis on combined data
    combined_confidence = np.abs(combined_predictions - combined_lines)

    # Find optimal threshold
    print(f"\n{'='*80}")
    print("SELECTIVE BETTING ANALYSIS")
    print(f"{'='*80}")

    print(f"\nTesting different confidence thresholds:")
    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'# Bets':<12} {'ROI':<12} {'Status':<12}")
    print("-" * 70)

    # Store detailed results for range analysis
    threshold_results = []

    for threshold in range(20, 125, 5):  # Test every 5 yards from 20 to 120
        mask = combined_confidence >= threshold
        if mask.sum() >= 20:  # Minimum 20 bets to be meaningful
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

            # Print key thresholds
            if threshold in [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
                print(f"{threshold:>3} yards     {filtered_accuracy*100:>6.1f}%      {filtered_total:>6}       {filtered_roi:>+6.2f}%      {status}")

    # Identify profitable ranges
    print(f"\n{'='*80}")
    print("OPTIMAL THRESHOLD RANGES")
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

                print(f"Range {idx}: {min_threshold}-{max_threshold} yards")
                print(f"  Avg Accuracy: {avg_accuracy*100:.1f}%")
                print(f"  Avg ROI: {avg_roi:+.2f}%")
                print(f"  Bets (at {min_threshold} yards): {total_bets_range}")

                # Show best performing threshold in this range
                best_in_range = max(range_data, key=lambda x: x['roi'])
                print(f"  Peak: {best_in_range['threshold']} yards ({best_in_range['accuracy']*100:.1f}% acc, {best_in_range['roi']:+.2f}% ROI, {best_in_range['bets']} bets)")
                print()

            # Strategy recommendations
            print(f"{'='*80}")
            print("STRATEGY RECOMMENDATIONS")
            print(f"{'='*80}")

            # Find the main profitable range (usually the largest)
            main_range = max(ranges, key=lambda r: len(r))
            min_t = main_range[0]['threshold']
            max_t = main_range[-1]['threshold']

            # Conservative: higher threshold, lower volume, higher accuracy
            conservative_data = main_range[-1] if len(main_range) > 1 else main_range[0]

            # Aggressive: lower threshold, higher volume, still profitable
            aggressive_data = main_range[0]

            # Balanced: middle of the range
            mid_idx = len(main_range) // 2
            balanced_data = main_range[mid_idx]

            print(f"\n1. CONSERVATIVE (High Accuracy, Low Volume)")
            print(f"   Threshold: {conservative_data['threshold']}+ yards")
            print(f"   Expected: {conservative_data['accuracy']*100:.1f}% accuracy, {conservative_data['roi']:+.2f}% ROI")
            print(f"   Volume: ~{conservative_data['bets']} bets/2 years (~{conservative_data['bets']/34:.1f} bets/week)")

            print(f"\n2. BALANCED (Good Accuracy, Moderate Volume)")
            print(f"   Threshold: {balanced_data['threshold']}+ yards")
            print(f"   Expected: {balanced_data['accuracy']*100:.1f}% accuracy, {balanced_data['roi']:+.2f}% ROI")
            print(f"   Volume: ~{balanced_data['bets']} bets/2 years (~{balanced_data['bets']/34:.1f} bets/week)")

            print(f"\n3. AGGRESSIVE (More Bets, Still Profitable)")
            print(f"   Threshold: {aggressive_data['threshold']}+ yards")
            print(f"   Expected: {aggressive_data['accuracy']*100:.1f}% accuracy, {aggressive_data['roi']:+.2f}% ROI")
            print(f"   Volume: ~{aggressive_data['bets']} bets/2 years (~{aggressive_data['bets']/34:.1f} bets/week)")

            print(f"\n💡 Recommendation: Start with BALANCED strategy ({balanced_data['threshold']}+ yards)")
            print(f"   This gives you ~{balanced_data['bets']/34:.1f} bets/week with strong profitability.")
            print(f"   Once validated in live betting, consider AGGRESSIVE for more volume.")
        else:
            print("\n❌ No profitable threshold ranges found.")
            print("   Model needs improvement before live betting.")
    else:
        print("\n❌ Insufficient data for range analysis.")

    # Year-over-year comparison
    print(f"\n{'='*80}")
    print("YEAR-OVER-YEAR COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Metric':<20} {'2023':<15} {'2024':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Test Examples':<20} {results_2023['test_examples']:<15,} {results_2024['test_examples']:<15,} {results_2024['test_examples'] - results_2023['test_examples']:+}")
    print(f"{'MAE':<20} {results_2023['mae']:<15.2f} {results_2024['mae']:<15.2f} {results_2024['mae'] - results_2023['mae']:+.2f}")
    print(f"{'RMSE':<20} {results_2023['rmse']:<15.2f} {results_2024['rmse']:<15.2f} {results_2024['rmse'] - results_2023['rmse']:+.2f}")
    print(f"{'Accuracy':<20} {results_2023['accuracy']*100:<15.1f}% {results_2024['accuracy']*100:<15.1f}% {(results_2024['accuracy'] - results_2023['accuracy'])*100:+.1f}%")
    print(f"{'ROI':<20} {results_2023['roi']:<15.2f}% {results_2024['roi']:<15.2f}% {results_2024['roi'] - results_2023['roi']:+.2f}%")

    # Summary
    print(f"\n{'='*80}")
    print("KEY TAKEAWAYS")
    print(f"{'='*80}")

    consistency = abs(results_2023['accuracy'] - results_2024['accuracy'])

    print(f"\n1. CONSISTENCY:")
    if consistency < 0.03:  # Within 3%
        print(f"   Model performance is CONSISTENT across years")
        print(f"   2023: {results_2023['accuracy']*100:.1f}%, 2024: {results_2024['accuracy']*100:.1f}% (diff: {consistency*100:.1f}%)")
    else:
        print(f"   Model performance VARIES between years")
        print(f"   2023: {results_2023['accuracy']*100:.1f}%, 2024: {results_2024['accuracy']*100:.1f}% (diff: {consistency*100:.1f}%)")

    print(f"\n2. OVERALL PERFORMANCE:")
    if combined_accuracy > 0.524:
        print(f"   PROFITABLE: {combined_accuracy*100:.1f}% accuracy (>{52.4}% break-even)")
        print(f"   Estimated ROI: {combined_roi:+.2f}% on {combined_total:,} bets")
    else:
        print(f"   NOT PROFITABLE: {combined_accuracy*100:.1f}% accuracy (<{52.4}% break-even)")
        print(f"   Need {int((0.524 * combined_total) - combined_wins)} more wins to break even")

    print(f"\n3. RECOMMENDATION:")

    # Find best threshold
    best_threshold = None
    best_roi = -float('inf')
    for threshold in range(30, 120, 5):
        mask = combined_confidence >= threshold
        if mask.sum() >= 50:  # Minimum 50 bets
            filtered_accuracy = combined_correct[mask].sum() / mask.sum()
            if filtered_accuracy > 0.524:
                filtered_wins = combined_correct[mask].sum()
                filtered_total = mask.sum()
                filtered_roi = ((filtered_wins * 210 - filtered_total * 110) / (filtered_total * 110)) * 100
                if filtered_roi > best_roi:
                    best_roi = filtered_roi
                    best_threshold = threshold
                    best_accuracy = filtered_accuracy
                    best_bets = filtered_total

    if best_threshold:
        print(f"   Use SELECTIVE BETTING with {best_threshold} yard threshold")
        print(f"   Expected: {best_accuracy*100:.1f}% accuracy on {best_bets} bets")
        print(f"   Projected ROI: {best_roi:+.2f}%")
    else:
        print(f"   Current model does NOT achieve profitability")
        print(f"   Need additional features or model improvements")

    print(f"\n{'='*80}")

    return {
        '2023': results_2023,
        '2024': results_2024,
        'combined': {
            'accuracy': combined_accuracy,
            'roi': combined_roi,
            'mae': combined_mae,
            'total_bets': combined_total
        }
    }


if __name__ == "__main__":
    results = main()
