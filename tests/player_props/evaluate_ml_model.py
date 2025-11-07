"""
ML Model Evaluation Against Actual Outcomes

Evaluates the trained ML ensemble using walk-forward validation to simulate
real-world betting performance. Tests on 2023 data (most recent year).

Metrics:
- Overall accuracy (% correct over/under predictions)
- Accuracy by confidence level
- Mean Absolute Error (MAE)
- Calibration analysis
- ROI simulation

Usage:
    python scripts/evaluate_ml_model.py
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

def evaluate_model(test_year: int = 2023):
    """
    Evaluate ML model on a held-out test year.

    Args:
        test_year: Year to use for testing (default: 2023)
    """

    print("\n" + "="*80)
    print(f"ML MODEL EVALUATION - Test Year: {test_year}")
    print("="*80)

    # Load full training data
    training_data_path = Path(CACHE_DIR) / "ml_training_data" / "passing_yards_2015_2023.parquet"

    if not training_data_path.exists():
        print(f"\nERROR: Training data not found at {training_data_path}")
        return

    print(f"\nLoading training data...")
    full_df = pl.read_parquet(training_data_path)

    # Split into train/test by year
    train_df = full_df.filter(pl.col('year') < test_year)
    test_df = full_df.filter(pl.col('year') == test_year)

    print(f"\nDataset Split:")
    print(f"  Train: {len(train_df):,} examples ({train_df['year'].min()}-{train_df['year'].max()})")
    print(f"  Test:  {len(test_df):,} examples ({test_year})")

    # Train model on pre-test-year data
    print(f"\n{'='*80}")
    print(f"Training Model on {train_df['year'].min()}-{train_df['year'].max()} Data")
    print(f"{'='*80}")

    ensemble = PropEnsembleModel(prop_type='passing_yards')
    ensemble.train(train_df, n_splits=3, verbose=False)

    print(f"\nModel trained. Generating predictions on {test_year} data...")

    # Generate predictions
    predictions = ensemble.predict_batch(test_df)
    actuals = test_df['target'].to_numpy()

    # Calculate basic metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    print(f"\n{'='*80}")
    print("PREDICTION ACCURACY")
    print(f"{'='*80}")
    print(f"Mean Absolute Error (MAE):  {mae:.2f} yards")
    print(f"Root Mean Squared Error:    {rmse:.2f} yards")
    print(f"Mean Actual:                {actuals.mean():.2f} yards")
    print(f"Mean Predicted:             {predictions.mean():.2f} yards")

    # Simulate betting lines evaluation
    # Assume typical betting line is set at actual - 5 to +5 yards (random noise)
    print(f"\n{'='*80}")
    print("SIMULATED BETTING PERFORMANCE")
    print(f"{'='*80}")
    print("\nSimulating betting lines as: actual ± random(0, 10) yards")
    print("(In reality, bookmakers set lines, but this simulates market efficiency)")

    np.random.seed(42)
    simulated_lines = actuals + np.random.uniform(-10, 10, size=len(actuals))

    # Binary prediction: does ML predict over or under the line?
    ml_predicts_over = predictions > simulated_lines
    actual_was_over = actuals > simulated_lines

    # Calculate accuracy
    correct_predictions = ml_predicts_over == actual_was_over
    accuracy = correct_predictions.sum() / len(correct_predictions)

    print(f"\nOverall Accuracy: {accuracy*100:.1f}%")
    print(f"Break-even: 52.4% (with -110 odds)")
    print(f"Edge: {(accuracy - 0.524)*100:+.1f}%")

    if accuracy > 0.524:
        print(f"\nRESULT: ML predictions BEAT break-even threshold!")
    else:
        print(f"\nRESULT: ML predictions below break-even (not profitable)")

    # Accuracy by confidence level
    print(f"\n{'='*80}")
    print("ACCURACY BY CONFIDENCE LEVEL")
    print(f"{'='*80}")
    print("\nConfidence = abs(prediction - line)")

    confidence = np.abs(predictions - simulated_lines)

    # Quartiles
    q1, q2, q3 = np.percentile(confidence, [25, 50, 75])

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
            print(f"Level {level} (avg edge: {avg_confidence:.1f} yards): "
                  f"{level_accuracy*100:.1f}% ({mask.sum()} bets)")

    # Calibration analysis
    print(f"\n{'='*80}")
    print("ERROR DISTRIBUTION")
    print(f"{'='*80}")

    errors = predictions - actuals

    print(f"Mean Error (bias):          {errors.mean():.2f} yards")
    print(f"Std Dev of Errors:          {errors.std():.2f} yards")
    print(f"Median Absolute Error:      {np.median(np.abs(errors)):.2f} yards")
    print(f"\nError Percentiles:")
    print(f"  5th:   {np.percentile(errors, 5):.1f} yards")
    print(f"  25th:  {np.percentile(errors, 25):.1f} yards")
    print(f"  50th:  {np.percentile(errors, 50):.1f} yards")
    print(f"  75th:  {np.percentile(errors, 75):.1f} yards")
    print(f"  95th:  {np.percentile(errors, 95):.1f} yards")

    # ROI simulation
    print(f"\n{'='*80}")
    print("ROI SIMULATION (if betting on all props)")
    print(f"{'='*80}")

    total_bets = len(correct_predictions)
    wins = correct_predictions.sum()
    losses = total_bets - wins

    # Assuming -110 odds (risk $110 to win $100)
    stake_per_bet = 110
    profit_per_win = 100

    total_wagered = total_bets * stake_per_bet
    gross_winnings = wins * (stake_per_bet + profit_per_win)  # Get stake back + profit
    gross_losses = losses * stake_per_bet
    net_profit = gross_winnings - total_wagered
    roi = (net_profit / total_wagered) * 100

    print(f"Total Bets:        {total_bets}")
    print(f"Wins:              {wins} ({accuracy*100:.1f}%)")
    print(f"Losses:            {losses} ({(1-accuracy)*100:.1f}%)")
    print(f"\nBetting -110 odds (standard):")
    print(f"Total Wagered:     ${total_wagered:,}")
    print(f"Gross Returns:     ${gross_winnings:,}")
    print(f"Net Profit/Loss:   ${net_profit:+,}")
    print(f"ROI:               {roi:+.2f}%")

    # Sample predictions
    print(f"\n{'='*80}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*80}")

    sample_indices = np.random.choice(len(test_df), size=min(10, len(test_df)), replace=False)

    print(f"\n{'Player':<20} {'Predicted':<12} {'Actual':<12} {'Line':<12} {'Result':<10}")
    print("-" * 80)

    # Convert to list for easier indexing
    player_ids = test_df['player_id'].to_list()

    for idx in sample_indices:
        pred = predictions[idx]
        actual = actuals[idx]
        line = simulated_lines[idx]

        # ML prediction vs line
        ml_pick = "OVER" if pred > line else "UNDER"
        # Actual result vs line
        actual_result = "OVER" if actual > line else "UNDER"
        # Was prediction correct?
        result = "WIN" if ml_pick == actual_result else "LOSS"

        # Get player_id (simplified display)
        player_id = player_ids[idx]
        player_display = player_id[-6:] if len(player_id) > 6 else player_id

        print(f"{player_display:<20} {pred:>6.1f} ({ml_pick:<5}) {actual:>6.1f} ({actual_result:<5}) {line:>6.1f}        {result:<10}")

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")

    return {
        'accuracy': accuracy,
        'mae': mae,
        'rmse': rmse,
        'roi': roi,
        'total_bets': total_bets,
        'wins': wins
    }


if __name__ == "__main__":
    results = evaluate_model(test_year=2023)

    print("\n" + "="*80)
    print("KEY TAKEAWAY")
    print("="*80)

    if results['accuracy'] > 0.524:
        print(f"\nThe ML model achieves {results['accuracy']*100:.1f}% accuracy,")
        print(f"which is {(results['accuracy'] - 0.524)*100:.1f}% above the 52.4% break-even threshold.")
        print(f"\nEstimated ROI: {results['roi']:+.2f}%")
        print("\nThis suggests the model has potential for profitable betting.")
    else:
        print(f"\nThe ML model achieves {results['accuracy']*100:.1f}% accuracy,")
        print(f"which is {(results['accuracy'] - 0.524)*100:.1f}% below the 52.4% break-even threshold.")
        print(f"\nEstimated ROI: {results['roi']:+.2f}%")
        print("\nMore work needed to achieve profitability.")

    print("\n" + "="*80)
