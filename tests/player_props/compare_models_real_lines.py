"""
Compare Full vs High-Volume WR Models on Real Betting Lines

Loads both models and evaluates them on 2024 test data with real prop lines.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import numpy as np
import json
from modules.ml_ensemble import PropEnsembleModel
from modules.constants import CACHE_DIR

def load_real_lines(seasons=[2024]):
    """Load real betting lines from cache/player_props."""
    props_dir = Path(CACHE_DIR) / "player_props"
    lines_dict = {}

    for season in seasons:
        season_dir = props_dir / str(season)
        if not season_dir.exists():
            continue

        for player_dir in season_dir.iterdir():
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
                        key = (gsis_id, season, week_num)
                        lines_dict[key] = recv_yards_data['line']
                except:
                    continue

    return lines_dict

def evaluate_model_on_real_lines(model, test_df, real_lines_dict, model_name):
    """Evaluate a model against real betting lines."""
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*80}")

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

    if len(lines) == 0:
        print("ERROR: No real lines matched!")
        return None

    lines = np.array(lines)
    matched_indices = np.array(matched_indices)

    # Get predictions for matched examples
    predictions = model.predict_batch(test_df)
    predictions_matched = predictions[matched_indices]
    actuals_matched = test_df[matched_indices].select('target').to_numpy().flatten()

    print(f"Matched examples: {len(lines)} with real prop lines")

    # Calculate MAE
    mae = np.mean(np.abs(predictions_matched - actuals_matched))
    print(f"MAE: {mae:.2f} yards")

    # Binary predictions (Over/Under)
    ml_over = predictions_matched > lines
    actual_over = actuals_matched > lines
    correct = ml_over == actual_over

    accuracy = correct.sum() / len(correct)

    # ROI Calculation
    wins = correct.sum()
    losses = len(correct) - wins
    breakeven_rate = 0.524

    total_wagered = len(correct) * 110
    gross_winnings = wins * 210
    net_profit = gross_winnings - total_wagered
    roi = (net_profit / total_wagered) * 100

    print(f"\nBinary Prediction Accuracy (Over/Under):")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Win Rate: {accuracy:.1%} (breakeven: {breakeven_rate:.1%})")
    print(f"  Record: {wins}-{losses}")
    print(f"  Net Profit: ${net_profit:,.0f} (on ${total_wagered:,.0f} wagered)")
    print(f"  ROI: {roi:+.2f}%")

    # Confidence-based analysis
    confidence = np.abs(predictions_matched - lines)

    print(f"\nConfidence-Based Performance:")
    for threshold in [5, 10, 15, 20]:
        mask = confidence >= threshold
        if mask.sum() > 0:
            high_conf_accuracy = correct[mask].sum() / mask.sum()
            high_conf_count = mask.sum()

            hc_wins = correct[mask].sum()
            hc_wagered = mask.sum() * 110
            hc_winnings = hc_wins * 210
            hc_profit = hc_winnings - hc_wagered
            hc_roi = (hc_profit / hc_wagered) * 100 if hc_wagered > 0 else 0

            print(f"  {threshold}+ yards: {high_conf_accuracy:.1%} ({high_conf_count} bets, ROI: {hc_roi:+.2f}%)")

    return {
        'accuracy': accuracy,
        'mae': mae,
        'roi': roi,
        'wins': wins,
        'losses': losses,
        'matched_examples': len(lines)
    }

def main():
    print("="*80)
    print("MODEL COMPARISON: FULL vs HIGH-VOLUME WR MODELS")
    print("="*80)
    print("Testing on 2024 data with REAL betting lines")

    # Load real betting lines
    print("\nLoading real betting lines (2024, friday)...")
    real_lines = load_real_lines(seasons=[2024])
    print(f"Loaded {len(real_lines)} real betting lines")

    # Load test data (full dataset)
    full_df = pl.read_parquet(Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2009_2024.parquet")
    test_2024_full = full_df.filter(pl.col('season') == 2024)

    # Load test data (high-volume only)
    high_vol_df = pl.read_parquet(Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2009_2024_high_volume.parquet")
    test_2024_high_vol = high_vol_df.filter(pl.col('season') == 2024)

    print(f"\nTest data loaded:")
    print(f"  Full dataset 2024: {len(test_2024_full)} examples")
    print(f"  High-volume 2024: {len(test_2024_high_vol)} examples")

    # Load models
    print(f"\nLoading models...")
    model_full = PropEnsembleModel(prop_type='receiving_yards_wr')
    model_full.load(str(Path(CACHE_DIR) / "ml_models" / "receiving_yards_wr_ensemble.pkl"))
    print(f"  Loaded full dataset model")

    model_high_vol = PropEnsembleModel(prop_type='receiving_yards_wr')
    model_high_vol.load(str(Path(CACHE_DIR) / "ml_models" / "receiving_yards_wr_ensemble_high_volume.pkl"))
    print(f"  Loaded high-volume model")

    # Evaluate full model on full test data
    results_full = evaluate_model_on_real_lines(
        model_full,
        test_2024_full,
        real_lines,
        "FULL DATASET MODEL (trained on all WRs)"
    )

    # Evaluate high-volume model on high-volume test data
    results_high_vol = evaluate_model_on_real_lines(
        model_high_vol,
        test_2024_high_vol,
        real_lines,
        "HIGH-VOLUME MODEL (trained on >=4.5 targets/game WRs only)"
    )

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")

    if results_full and results_high_vol:
        print(f"\nFull Model:")
        print(f"  Accuracy: {results_full['accuracy']:.1%}")
        print(f"  ROI: {results_full['roi']:+.2f}%")
        print(f"  Record: {results_full['wins']}-{results_full['losses']}")
        print(f"  MAE: {results_full['mae']:.2f} yards")

        print(f"\nHigh-Volume Model:")
        print(f"  Accuracy: {results_high_vol['accuracy']:.1%}")
        print(f"  ROI: {results_high_vol['roi']:+.2f}%")
        print(f"  Record: {results_high_vol['wins']}-{results_high_vol['losses']}")
        print(f"  MAE: {results_high_vol['mae']:.2f} yards")

        print(f"\nImprovement (High-Volume - Full):")
        print(f"  Accuracy: {(results_high_vol['accuracy'] - results_full['accuracy'])*100:+.1f} percentage points")
        print(f"  ROI: {results_high_vol['roi'] - results_full['roi']:+.2f}% points")
        print(f"  MAE: {results_high_vol['mae'] - results_full['mae']:+.2f} yards")

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
