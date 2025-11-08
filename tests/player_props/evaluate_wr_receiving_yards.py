"""
WR Receiving Yards Model - Full Evaluation Report

Trains and evaluates the WR receiving yards ensemble model following the .md plan:
1. Train on 2009-2022 data
2. Test on 2023-2024 held-out data
3. Compare to baseline (simple average)
4. Report accuracy, ROI, and betting performance
5. Use REAL betting lines from cache/player_props

Usage:
    python tests/player_props/evaluate_wr_receiving_yards.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import numpy as np
import json
from modules.ml_ensemble import PropEnsembleModel
from modules.constants import CACHE_DIR
from datetime import datetime

def load_real_betting_lines(seasons=[2023, 2024], day='friday'):
    """
    Load real betting lines from cache/player_props.

    Args:
        seasons: List of seasons to load (default [2023, 2024])
        day: Which day's lines to use - 'friday' (closing) or 'tuesday' (opening)

    Returns:
        Dict mapping (gsis_id, season, week) -> line value
    """
    print(f"\nLoading real betting lines ({day})...")
    props_dir = Path(CACHE_DIR) / "player_props"
    lines_dict = {}

    for season in seasons:
        season_dir = props_dir / str(season)
        if not season_dir.exists():
            print(f"  Warning: {season_dir} does not exist")
            continue

        # Iterate through all player directories
        player_dirs = [d for d in season_dir.iterdir() if d.is_dir()]

        for player_dir in player_dirs:
            # Iterate through all week directories
            week_dirs = [d for d in player_dir.iterdir() if d.is_dir() and d.name.startswith('week')]

            for week_dir in week_dirs:
                # Try to load the specified day's file
                props_file = week_dir / f"{day}.json"
                if not props_file.exists():
                    continue

                try:
                    with open(props_file, 'r') as f:
                        data = json.load(f)

                    # Extract relevant info
                    gsis_id = data.get('gsis_id')
                    week_num = data['metadata']['week']
                    position = data.get('position')

                    # Only process WR receiving yards
                    if position != 'WR':
                        continue

                    # Extract receiving yards line
                    props = data.get('props', {})
                    recv_yards_data = props.get('Receiving Yards')

                    if recv_yards_data and 'line' in recv_yards_data:
                        line = recv_yards_data['line']
                        key = (gsis_id, season, week_num)
                        lines_dict[key] = line

                except (json.JSONDecodeError, KeyError) as e:
                    # Skip malformed files
                    continue

    print(f"  Loaded {len(lines_dict):,} real betting lines")
    return lines_dict

def calculate_baseline_accuracy(test_df):
    """Calculate baseline accuracy using simple weighted average."""
    predictions = test_df['weighted_avg'].to_numpy()
    actuals = test_df['target'].to_numpy()

    # Simulate lines
    np.random.seed(42)
    simulated_lines = actuals + np.random.uniform(-10, 10, size=len(actuals))

    # Binary predictions
    baseline_over = predictions > simulated_lines
    actual_over = actuals > simulated_lines
    correct = baseline_over == actual_over

    return correct.sum() / len(correct)

def evaluate_model(model, test_df, year_label, real_lines_dict=None):
    """
    Evaluate model on test data.

    Args:
        model: Trained ensemble model
        test_df: Test dataframe
        year_label: Label for output (e.g., "2023")
        real_lines_dict: Optional dict of (gsis_id, season, week) -> line
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING ON {year_label}")
    print(f"{'='*80}")
    print(f"Test examples: {len(test_df):,}")

    # Generate predictions
    predictions = model.predict_batch(test_df)
    actuals = test_df['target'].to_numpy()

    # Calculate MAE/RMSE
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    print(f"\nPrediction Accuracy:")
    print(f"  MAE:  {mae:.2f} yards")
    print(f"  RMSE: {rmse:.2f} yards")

    # Try to use real lines if available
    use_real_lines = real_lines_dict is not None

    if use_real_lines:
        print(f"\nMatching to real betting lines...")
        # Match test data to real lines
        lines = []
        matched_indices = []

        for idx in range(len(test_df)):
            row = test_df[idx]
            player_id = row['player_id'][0] if hasattr(row['player_id'], '__getitem__') else row['player_id']
            season = row['season'][0] if hasattr(row['season'], '__getitem__') else row['season']
            week = row['week'][0] if hasattr(row['week'], '__getitem__') else row['week']

            key = (player_id, season, week)
            if key in real_lines_dict:
                lines.append(real_lines_dict[key])
                matched_indices.append(idx)

        if len(lines) > 0:
            lines = np.array(lines)
            matched_indices = np.array(matched_indices)

            # Filter to matched examples only
            predictions_matched = predictions[matched_indices]
            actuals_matched = actuals[matched_indices]

            match_rate = len(lines) / len(test_df)
            print(f"  Matched {len(lines):,} / {len(test_df):,} examples ({match_rate:.1%})")
            print(f"  Using REAL betting lines for evaluation")
        else:
            print(f"  WARNING: No real lines matched! Falling back to simulated lines")
            use_real_lines = False

    if not use_real_lines:
        # Simulate betting lines (fallback)
        print(f"\nUsing SIMULATED betting lines (no real data available)")
        np.random.seed(42 + hash(year_label) % 1000)
        lines = actuals + np.random.uniform(-10, 10, size=len(actuals))
        predictions_matched = predictions
        actuals_matched = actuals
        matched_indices = np.arange(len(actuals))

    # Binary predictions (Over/Under)
    ml_over = predictions_matched > lines
    actual_over = actuals_matched > lines
    correct = ml_over == actual_over

    accuracy = correct.sum() / len(correct)

    # Baseline accuracy
    baseline_acc = calculate_baseline_accuracy(test_df)

    print(f"\nBinary Prediction Accuracy (Over/Under):")
    print(f"  ML Model:  {accuracy:.1%}")
    print(f"  Baseline:  {baseline_acc:.1%}")
    print(f"  Improvement: {(accuracy - baseline_acc)*100:+.1f} percentage points")

    # ROI Calculation (assume -110 odds, need 52.4% to break even)
    wins = correct.sum()
    losses = len(correct) - wins
    breakeven_rate = 0.524

    # Calculate profit assuming $110 wagered per bet
    total_wagered = len(correct) * 110
    gross_winnings = wins * 210  # Win $100 + get stake back
    net_profit = gross_winnings - total_wagered
    roi = (net_profit / total_wagered) * 100

    print(f"\nROI Analysis:")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {accuracy:.1%} (breakeven: {breakeven_rate:.1%})")
    print(f"  Net Profit: ${net_profit:,.0f} (on ${total_wagered:,.0f} wagered)")
    print(f"  ROI: {roi:+.2f}%")

    # Confidence-based betting analysis
    confidence = np.abs(predictions_matched - lines)

    print(f"\nConfidence-Based Performance:")
    for threshold in [5, 10, 15, 20]:
        mask = confidence >= threshold
        if mask.sum() > 0:
            high_conf_accuracy = correct[mask].sum() / mask.sum()
            high_conf_count = mask.sum()
            high_conf_pct = (high_conf_count / len(correct)) * 100

            # ROI for this threshold
            hc_wins = correct[mask].sum()
            hc_losses = mask.sum() - hc_wins
            hc_wagered = mask.sum() * 110
            hc_winnings = hc_wins * 210
            hc_profit = hc_winnings - hc_wagered
            hc_roi = (hc_profit / hc_wagered) * 100 if hc_wagered > 0 else 0

            print(f"  {threshold}+ yards confidence: {high_conf_accuracy:.1%} ({high_conf_count} bets, {high_conf_pct:.1f}% of total, ROI: {hc_roi:+.2f}%)")

    return {
        'accuracy': accuracy,
        'baseline_accuracy': baseline_acc,
        'mae': mae,
        'rmse': rmse,
        'roi': roi,
        'wins': wins,
        'losses': losses,
        'examples': len(test_df),
        'real_lines_used': use_real_lines,
        'match_rate': len(lines) / len(test_df) if use_real_lines else 0.0
    }

def main():
    print("="*80)
    print("WR RECEIVING YARDS MODEL - FULL EVALUATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load training data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    training_file = Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2009_2024.parquet"
    print(f"Loading from: {training_file}")

    full_df = pl.read_parquet(training_file)
    print(f"Total examples: {len(full_df):,}")
    print(f"Features: {len(full_df.columns)}")

    # Split train/test
    train_df = full_df.filter(pl.col('season') <= 2022)
    test_2023 = full_df.filter(pl.col('season') == 2023)
    test_2024 = full_df.filter(pl.col('season') == 2024)
    test_combined = full_df.filter(pl.col('season').is_in([2023, 2024]))

    print(f"\nData Split:")
    print(f"  Train (2009-2022): {len(train_df):,} examples")
    print(f"  Test 2023: {len(test_2023):,} examples")
    print(f"  Test 2024: {len(test_2024):,} examples")
    print(f"  Test Combined: {len(test_combined):,} examples")

    # Train model
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)

    model = PropEnsembleModel(prop_type='receiving_yards_wr')
    model.train(train_df)

    # Save model
    model_path = Path(CACHE_DIR) / "ml_models" / "receiving_yards_wr_ensemble.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Load real betting lines
    real_lines = load_real_betting_lines(seasons=[2023, 2024], day='friday')

    # Evaluate on test sets
    results_2023 = evaluate_model(model, test_2023, "2023", real_lines_dict=real_lines)
    results_2024 = evaluate_model(model, test_2024, "2024", real_lines_dict=real_lines)
    results_combined = evaluate_model(model, test_combined, "2023-2024 Combined", real_lines_dict=real_lines)

    # Summary report
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    # Report on real lines usage
    if results_combined['real_lines_used']:
        print(f"\nREAL BETTING LINES USED")
        print(f"  Match Rate: {results_combined['match_rate']:.1%}")
        print(f"  Source: Friday closing lines from cache/player_props")
    else:
        print(f"\nSIMULATED BETTING LINES USED (no real data available)")

    print(f"\n2023 Results:")
    print(f"  Accuracy: {results_2023['accuracy']:.1%} (baseline: {results_2023['baseline_accuracy']:.1%})")
    print(f"  MAE: {results_2023['mae']:.2f} yards")
    print(f"  ROI: {results_2023['roi']:+.2f}%")
    if results_2023['real_lines_used']:
        print(f"  Match Rate: {results_2023['match_rate']:.1%}")

    print(f"\n2024 Results:")
    print(f"  Accuracy: {results_2024['accuracy']:.1%} (baseline: {results_2024['baseline_accuracy']:.1%})")
    print(f"  MAE: {results_2024['mae']:.2f} yards")
    print(f"  ROI: {results_2024['roi']:+.2f}%")
    if results_2024['real_lines_used']:
        print(f"  Match Rate: {results_2024['match_rate']:.1%}")

    print(f"\nCombined 2023-2024 Results:")
    print(f"  Accuracy: {results_combined['accuracy']:.1%} (baseline: {results_combined['baseline_accuracy']:.1%})")
    print(f"  Improvement: {(results_combined['accuracy'] - results_combined['baseline_accuracy'])*100:+.1f} pp")
    print(f"  MAE: {results_combined['mae']:.2f} yards")
    print(f"  ROI: {results_combined['roi']:+.2f}%")
    print(f"  Record: {results_combined['wins']}-{results_combined['losses']}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
