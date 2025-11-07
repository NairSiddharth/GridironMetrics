"""
Compare All Prop Type Models - Comprehensive Evaluation

Evaluates all trained prop type models and generates a comparison report:
- Training performance (MAE, model weights)
- Test performance (2023 & 2024)
- Betting profitability
- Optimal thresholds
- Side-by-side comparison

Usage:
    python scripts/compare_all_prop_types.py

Output:
    - Console summary
    - ML_PROP_COMPARISON.md (detailed markdown report)
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
from modules.logger import get_logger

logger = get_logger(__name__)

ALL_PROP_TYPES = [
    'passing_yards',
    'passing_tds',
    'rushing_yards',
    'rushing_tds',
    'receptions',
    'receiving_yards',
    'receiving_tds'
]


def check_available_models():
    """Check which prop type models have been trained."""
    available = []
    for prop_type in ALL_PROP_TYPES:
        model_path = Path(CACHE_DIR) / "ml_models" / prop_type / f"{prop_type}_ensemble.joblib"
        if model_path.exists():
            available.append(prop_type)
    return available


def evaluate_prop_type(prop_type: str, test_years: list = [2023, 2024]):
    """
    Evaluate a single prop type on test years.

    Returns:
        Dictionary with training and test performance metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {prop_type.upper()}")
    print(f"{'='*80}")

    # Load model
    ensemble = PropEnsembleModel(prop_type=prop_type)
    try:
        ensemble.load()
    except FileNotFoundError:
        print(f"ERROR: Model not found for {prop_type}")
        return None

    print(f"Model loaded from: cache/ml_models/{prop_type}/")
    print(f"\nModel Weights:")
    for model_name, weight in ensemble.weights.items():
        print(f"  {model_name:15s}: {weight:.4f}")

    # Load or generate test data
    training_data_path = Path(CACHE_DIR) / "ml_training_data" / f"{prop_type}_2015_2024.parquet"

    if not training_data_path.exists():
        print(f"\nGenerating test data for {prop_type}...")
        builder = TrainingDataBuilder()
        full_df = builder.build_training_dataset(
            start_year=2015,
            end_year=2024,
            prop_type=prop_type
        )
        if len(full_df) == 0:
            print(f"ERROR: No data generated for {prop_type}")
            return None
    else:
        full_df = pl.read_parquet(training_data_path)

    # Split by year
    test_dfs = {}
    for year in test_years:
        test_df = full_df.filter(pl.col('year') == year)
        if len(test_df) > 0:
            test_dfs[year] = test_df

    if not test_dfs:
        print(f"ERROR: No test data available for {test_years}")
        return None

    # Evaluate on each year
    year_results = {}

    for year, test_df in test_dfs.items():
        print(f"\n--- Testing on {year} ---")
        print(f"Test examples: {len(test_df):,}")

        # Generate predictions
        predictions = ensemble.predict_batch(test_df)
        actuals = test_df['target'].to_numpy()

        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        # Simulate betting
        np.random.seed(42 + year)
        simulated_lines = actuals + np.random.uniform(-10, 10, size=len(actuals))

        ml_predicts_over = predictions > simulated_lines
        actual_was_over = actuals > simulated_lines
        correct = ml_predicts_over == actual_was_over

        accuracy = correct.sum() / len(correct)

        # ROI
        wins = correct.sum()
        total = len(correct)
        roi = ((wins * 210 - total * 110) / (total * 110)) * 100

        print(f"Accuracy: {accuracy*100:.1f}%")
        print(f"ROI: {roi:+.2f}%")

        # Find optimal threshold
        confidence = np.abs(predictions - simulated_lines)
        best_threshold = None
        best_roi = -float('inf')
        best_accuracy = 0
        best_bets = 0

        for threshold in range(10, 150, 5):
            mask = confidence >= threshold
            if mask.sum() >= 30:  # Minimum 30 bets
                filtered_accuracy = correct[mask].sum() / mask.sum()
                if filtered_accuracy > 0.524:
                    filtered_wins = correct[mask].sum()
                    filtered_total = mask.sum()
                    filtered_roi = ((filtered_wins * 210 - filtered_total * 110) / (filtered_total * 110)) * 100
                    if filtered_roi > best_roi:
                        best_roi = filtered_roi
                        best_threshold = threshold
                        best_accuracy = filtered_accuracy
                        best_bets = filtered_total

        year_results[year] = {
            'examples': len(test_df),
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy,
            'roi': roi,
            'wins': wins,
            'total': total,
            'best_threshold': best_threshold,
            'best_accuracy': best_accuracy,
            'best_roi': best_roi,
            'best_bets': best_bets
        }

        if best_threshold:
            print(f"Best threshold: {best_threshold} (accuracy: {best_accuracy*100:.1f}%, ROI: {best_roi:+.2f}% on {best_bets} bets)")

    # Combined results
    if len(test_dfs) > 1:
        print(f"\n--- Combined {test_years} ---")
        combined_test = pl.concat(list(test_dfs.values()))
        combined_predictions = ensemble.predict_batch(combined_test)
        combined_actuals = combined_test['target'].to_numpy()

        combined_mae = np.mean(np.abs(combined_predictions - combined_actuals))

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

        print(f"Combined MAE: {combined_mae:.2f}")
        print(f"Combined Accuracy: {combined_accuracy*100:.1f}%")
        print(f"Combined ROI: {combined_roi:+.2f}%")

        # Find optimal threshold for combined
        combined_confidence = np.abs(combined_predictions - combined_lines)
        best_combined_threshold = None
        best_combined_roi = -float('inf')
        best_combined_accuracy = 0
        best_combined_bets = 0

        for threshold in range(10, 150, 5):
            mask = combined_confidence >= threshold
            if mask.sum() >= 50:
                filtered_accuracy = combined_correct[mask].sum() / mask.sum()
                if filtered_accuracy > 0.524:
                    filtered_wins = combined_correct[mask].sum()
                    filtered_total = mask.sum()
                    filtered_roi = ((filtered_wins * 210 - filtered_total * 110) / (filtered_total * 110)) * 100
                    if filtered_roi > best_combined_roi:
                        best_combined_roi = filtered_roi
                        best_combined_threshold = threshold
                        best_combined_accuracy = filtered_accuracy
                        best_combined_bets = filtered_total

        if best_combined_threshold:
            print(f"Best combined threshold: {best_combined_threshold} (accuracy: {best_combined_accuracy*100:.1f}%, ROI: {best_combined_roi:+.2f}% on {best_combined_bets} bets)")

    return {
        'prop_type': prop_type,
        'weights': ensemble.weights,
        'year_results': year_results,
        'combined': {
            'mae': combined_mae if len(test_dfs) > 1 else year_results[test_years[0]]['mae'],
            'accuracy': combined_accuracy if len(test_dfs) > 1 else year_results[test_years[0]]['accuracy'],
            'roi': combined_roi if len(test_dfs) > 1 else year_results[test_years[0]]['roi'],
            'best_threshold': best_combined_threshold if len(test_dfs) > 1 else year_results[test_years[0]]['best_threshold'],
            'best_accuracy': best_combined_accuracy if len(test_dfs) > 1 else year_results[test_years[0]]['best_accuracy'],
            'best_roi': best_combined_roi if len(test_dfs) > 1 else year_results[test_years[0]]['best_roi'],
            'best_bets': best_combined_bets if len(test_dfs) > 1 else year_results[test_years[0]]['best_bets']
        }
    }


def generate_markdown_report(results: dict):
    """Generate comprehensive markdown comparison report."""

    report = """# ML Prop Type Comparison Report

**Evaluation Date:** 2025-11-03
**Test Years:** 2023 & 2024
**Training Data:** 2015-2022
**Break-even Threshold:** 52.4% accuracy (with -110 odds)

---

## Executive Summary

This report compares trained ML ensemble models across different prop types.

### Trained Models

"""

    for prop_type, result in results.items():
        if result:
            status = "✅ PROFITABLE" if result['combined']['best_roi'] and result['combined']['best_roi'] > 0 else "❌ Unprofitable"
            report += f"- **{prop_type}**: {status}\n"

    report += "\n---\n\n## Detailed Comparison\n\n"
    report += f"| Prop Type | MAE | Overall Acc | Overall ROI | Best Threshold | Best Acc | Best ROI | Bets |\n"
    report += f"|-----------|-----|-------------|-------------|----------------|----------|----------|------|\n"

    for prop_type, result in results.items():
        if result:
            combined = result['combined']
            threshold_str = f"{combined['best_threshold']}" if combined['best_threshold'] else "N/A"
            best_acc_str = f"{combined['best_accuracy']*100:.1f}%" if combined['best_threshold'] else "N/A"
            best_roi_str = f"{combined['best_roi']:+.1f}%" if combined['best_threshold'] else "N/A"
            best_bets_str = f"{combined['best_bets']}" if combined['best_threshold'] else "N/A"

            report += f"| **{prop_type}** | {combined['mae']:.2f} | {combined['accuracy']*100:.1f}% | {combined['roi']:+.1f}% | {threshold_str} | {best_acc_str} | {best_roi_str} | {best_bets_str} |\n"

    report += "\n---\n\n## Individual Prop Type Details\n\n"

    for prop_type, result in results.items():
        if result:
            report += f"### {prop_type.upper()}\n\n"

            report += f"**Model Weights:**\n"
            for model_name, weight in result['weights'].items():
                report += f"- {model_name}: {weight:.4f}\n"

            report += f"\n**Year-by-Year Performance:**\n\n"
            report += f"| Year | Examples | MAE | Accuracy | ROI | Best Threshold |\n"
            report += f"|------|----------|-----|----------|-----|----------------|\n"

            for year, year_result in result['year_results'].items():
                threshold_str = f"{year_result['best_threshold']}" if year_result['best_threshold'] else "None"
                report += f"| {year} | {year_result['examples']:,} | {year_result['mae']:.2f} | {year_result['accuracy']*100:.1f}% | {year_result['roi']:+.1f}% | {threshold_str} |\n"

            report += f"\n**Combined Performance (2023 + 2024):**\n"
            combined = result['combined']
            report += f"- Overall Accuracy: **{combined['accuracy']*100:.1f}%**\n"
            report += f"- Overall ROI: **{combined['roi']:+.2f}%**\n"

            if combined['best_threshold']:
                report += f"\n**Optimal Selective Betting:**\n"
                report += f"- Threshold: **{combined['best_threshold']}** (|prediction - line|)\n"
                report += f"- Accuracy: **{combined['best_accuracy']*100:.1f}%** (>{52.4}% break-even)\n"
                report += f"- ROI: **{combined['best_roi']:+.2f}%**\n"
                report += f"- Number of bets: **{combined['best_bets']}** across 2 years\n"
                report += f"- Strategy: Only bet when model edge > {combined['best_threshold']}\n"
            else:
                report += f"\n❌ No profitable threshold found (model needs improvement)\n"

            report += f"\n---\n\n"

    report += f"""## Key Findings

### Best Performing Prop Types

"""

    # Sort by best ROI
    sorted_results = sorted(
        [(prop, res) for prop, res in results.items() if res and res['combined']['best_roi']],
        key=lambda x: x[1]['combined']['best_roi'] if x[1]['combined']['best_roi'] else -float('inf'),
        reverse=True
    )

    if sorted_results:
        for i, (prop_type, result) in enumerate(sorted_results[:3], 1):
            combined = result['combined']
            report += f"{i}. **{prop_type}**: {combined['best_accuracy']*100:.1f}% accuracy, {combined['best_roi']:+.2f}% ROI at {combined['best_threshold']} threshold\n"

    report += f"""

### Recommendations

"""

    profitable_props = [prop for prop, res in results.items() if res and res['combined']['best_roi'] and res['combined']['best_roi'] > 5]

    if profitable_props:
        report += f"**Implement Live Betting For:**\n"
        for prop_type in profitable_props:
            result = results[prop_type]
            combined = result['combined']
            report += f"- {prop_type}: Use {combined['best_threshold']} threshold (expected {combined['best_roi']:+.1f}% ROI)\n"
    else:
        report += f"**No prop types currently profitable enough for live betting.**\n"
        report += f"Consider:\n"
        report += f"- Feature engineering (weather, injuries, rest days)\n"
        report += f"- Hyperparameter tuning\n"
        report += f"- Collecting real betting lines for evaluation\n"

    report += f"""

---

**Model Files:**
- Training data: `cache/ml_training_data/{{prop_type}}_2015_2024.parquet`
- Models: `cache/ml_models/{{prop_type}}/{{prop_type}}_ensemble.joblib`
- Evaluation script: `scripts/compare_all_prop_types.py`
"""

    return report


def main():
    print("\n" + "="*80)
    print("ML PROP TYPE COMPARISON")
    print("="*80)

    # Check available models
    available = check_available_models()

    if not available:
        print("\nNo trained models found. Train models first using:")
        print("  python scripts/train_all_prop_types.py")
        return

    print(f"\nAvailable trained models: {', '.join(available)}")
    print(f"\nEvaluating {len(available)} prop types on 2023 & 2024 test data...")

    # Evaluate each prop type
    results = {}
    for prop_type in available:
        try:
            result = evaluate_prop_type(prop_type, test_years=[2023, 2024])
            if result:
                results[prop_type] = result
        except Exception as e:
            print(f"\nERROR evaluating {prop_type}: {e}")
            import traceback
            traceback.print_exc()
            results[prop_type] = None

    # Generate markdown report
    print(f"\n{'='*80}")
    print("GENERATING REPORT")
    print(f"{'='*80}")

    report = generate_markdown_report(results)

    # Save report
    report_path = Path("output/player_props/ML_PROP_COMPARISON.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Prop Type':<20} {'Overall Acc':<15} {'Best ROI':<15} {'Status':<15}")
    print("-" * 70)

    for prop_type, result in results.items():
        if result:
            combined = result['combined']
            status = "✅ PROFITABLE" if combined['best_roi'] and combined['best_roi'] > 0 else "❌ Below break-even"
            best_roi_str = f"{combined['best_roi']:+.1f}%" if combined['best_roi'] else "N/A"
            print(f"{prop_type:<20} {combined['accuracy']*100:<14.1f}% {best_roi_str:<15} {status:<15}")

    print(f"\n{'='*80}")
    print(f"Evaluation complete for {len(results)} prop types")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    results = main()
