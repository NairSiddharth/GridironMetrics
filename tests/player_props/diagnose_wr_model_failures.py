"""
Comprehensive Diagnostic Analysis: Why is the WR Receiving Yards Model Failing?

This script performs 6 diagnostic tests to identify the root cause of poor model performance:
1. Correlation analysis (are we copying lines?)
2. Error patterns (where do we fail?)
3. Baseline comparison (are lines just better?)
4. Feature redundancy (does weighted_avg dominate?)
5. Worst predictions (specific failure examples)
6. Information gap (what don't we have?)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import numpy as np
import json
from scipy import stats
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

def main():
    print("="*80)
    print("WR RECEIVING YARDS MODEL - COMPREHENSIVE DIAGNOSTIC ANALYSIS")
    print("="*80)
    print()

    # Load test data and model
    print("Loading 2024 test data and model...")
    test_df = pl.read_parquet(Path(CACHE_DIR) / "ml_training_data" / "receiving_yards_wr_2016_2024.parquet")
    test_df = test_df.filter(pl.col('year') == 2024)

    model = PropEnsembleModel(prop_type='receiving_yards_wr')
    model.load(str(Path(CACHE_DIR) / "ml_models" / "receiving_yards_wr_ensemble.pkl"))

    real_lines_dict = load_real_lines([2024])

    # Match test data to real lines
    lines = []
    matched_indices = []

    for idx in range(len(test_df)):
        row = test_df[idx]
        player_id = row['player_id'][0] if hasattr(row['player_id'], '__getitem__') else row['player_id']
        season = row['year'][0] if hasattr(row['year'], '__getitem__') else row['year']
        week = row['week'][0] if hasattr(row['week'], '__getitem__') else row['week']

        key = (player_id, season, week)
        if key in real_lines_dict:
            lines.append(real_lines_dict[key])
            matched_indices.append(idx)

    lines = np.array(lines)
    matched_indices = np.array(matched_indices)
    matched_df = test_df[matched_indices]

    # Get predictions and actuals
    predictions = model.predict_batch(test_df)[matched_indices]
    actuals = matched_df.select('target').to_numpy().flatten()

    # Extract key features for analysis
    weighted_avg = matched_df.select('weighted_avg').to_numpy().flatten()
    career_avg = matched_df.select('career_avg').to_numpy().flatten()
    is_home = matched_df.select('is_home').to_numpy().flatten()
    is_dome = matched_df.select('is_dome').to_numpy().flatten()
    vegas_total = matched_df.select('vegas_total').to_numpy().flatten()
    targets_season_avg = matched_df.select('targets_season_avg').to_numpy().flatten()

    print(f"Matched {len(lines)} examples with real betting lines")
    print()

    # ========================================================================
    # TEST 1: CORRELATION ANALYSIS
    # ========================================================================
    print("="*80)
    print("TEST 1: CORRELATION ANALYSIS - Are we just copying the lines?")
    print("="*80)
    print()

    corr_pred_line = np.corrcoef(predictions, lines)[0, 1]
    corr_pred_actual = np.corrcoef(predictions, actuals)[0, 1]
    corr_weighted_avg_line = np.corrcoef(weighted_avg, lines)[0, 1]
    corr_weighted_avg_pred = np.corrcoef(weighted_avg, predictions)[0, 1]
    corr_line_actual = np.corrcoef(lines, actuals)[0, 1]

    print(f"Correlation: Model Predictions vs Betting Lines:  {corr_pred_line:.4f}")
    print(f"Correlation: Model Predictions vs Actual Results: {corr_pred_actual:.4f}")
    print(f"Correlation: weighted_avg vs Betting Lines:       {corr_weighted_avg_line:.4f}")
    print(f"Correlation: weighted_avg vs Model Predictions:   {corr_weighted_avg_pred:.4f}")
    print(f"Correlation: Betting Lines vs Actual Results:     {corr_line_actual:.4f}")
    print()

    if corr_pred_line > 0.9:
        print("DIAGNOSIS: Model predictions are HIGHLY correlated with betting lines (>0.9)")
        print("           This suggests the model is largely copying the lines rather than")
        print("           providing independent signal.")
    elif corr_pred_line > 0.7:
        print("DIAGNOSIS: Model predictions are MODERATELY correlated with betting lines")
        print("           Some independence exists but still heavily influenced by line.")
    else:
        print("DIAGNOSIS: Model predictions have LOW correlation with betting lines")
        print("           Model is making independent predictions.")
    print()

    # ========================================================================
    # TEST 2: ERROR PATTERN ANALYSIS
    # ========================================================================
    print("="*80)
    print("TEST 2: ERROR PATTERN ANALYSIS - Where do we fail?")
    print("="*80)
    print()

    errors = predictions - actuals
    abs_errors = np.abs(errors)

    # Over/under prediction bias
    over_predictions = np.sum(errors > 0)
    under_predictions = np.sum(errors < 0)
    mean_error = np.mean(errors)

    print(f"Over-predictions: {over_predictions} ({100*over_predictions/len(errors):.1f}%)")
    print(f"Under-predictions: {under_predictions} ({100*under_predictions/len(errors):.1f}%)")
    print(f"Mean error (bias): {mean_error:.2f} yards")
    print()

    if abs(mean_error) > 2:
        print(f"DIAGNOSIS: Model has systematic {'OVER' if mean_error > 0 else 'UNDER'}-prediction bias")
    else:
        print("DIAGNOSIS: No significant prediction bias")
    print()

    # Errors by line value
    print("Errors by Line Value:")
    line_bins = [(0, 40), (40, 60), (60, 80), (80, 200)]
    for low, high in line_bins:
        mask = (lines >= low) & (lines < high)
        if np.sum(mask) > 0:
            bin_mae = np.mean(abs_errors[mask])
            bin_count = np.sum(mask)
            print(f"  {low}-{high} yards ({bin_count:3d} examples): MAE = {bin_mae:.2f}")
    print()

    # Errors by player volume (quartiles)
    print("Errors by Player Volume (targets/game):")
    volume_quartiles = np.percentile(targets_season_avg, [25, 50, 75])
    volume_bins = [
        (0, volume_quartiles[0], "Low volume"),
        (volume_quartiles[0], volume_quartiles[1], "Med-low volume"),
        (volume_quartiles[1], volume_quartiles[2], "Med-high volume"),
        (volume_quartiles[2], 20, "High volume")
    ]
    for low, high, label in volume_bins:
        mask = (targets_season_avg >= low) & (targets_season_avg < high)
        if np.sum(mask) > 0:
            bin_mae = np.mean(abs_errors[mask])
            bin_count = np.sum(mask)
            print(f"  {label:20s} ({bin_count:3d} examples): MAE = {bin_mae:.2f}")
    print()

    # Errors by game context
    print("Errors by Game Context:")
    print(f"  Home games ({np.sum(is_home==1):3d} examples): MAE = {np.mean(abs_errors[is_home==1]):.2f}")
    print(f"  Away games ({np.sum(is_home==0):3d} examples): MAE = {np.mean(abs_errors[is_home==0]):.2f}")
    print(f"  Dome games ({np.sum(is_dome==1):3d} examples): MAE = {np.mean(abs_errors[is_dome==1]):.2f}")
    print(f"  Outdoor    ({np.sum(is_dome==0):3d} examples): MAE = {np.mean(abs_errors[is_dome==0]):.2f}")

    # Vegas total context (if available)
    valid_vegas = ~np.isnan(vegas_total)
    if np.sum(valid_vegas) > 10:
        vegas_median = np.median(vegas_total[valid_vegas])
        high_total = (vegas_total >= vegas_median) & valid_vegas
        low_total = (vegas_total < vegas_median) & valid_vegas
        print(f"  High total ({np.sum(high_total):3d} examples): MAE = {np.mean(abs_errors[high_total]):.2f}")
        print(f"  Low total  ({np.sum(low_total):3d} examples): MAE = {np.mean(abs_errors[low_total]):.2f}")
    print()

    # ========================================================================
    # TEST 3: BASELINE COMPARISON
    # ========================================================================
    print("="*80)
    print("TEST 3: BASELINE COMPARISON - Are the lines just better?")
    print("="*80)
    print()

    model_mae = np.mean(abs_errors)
    line_mae = np.mean(np.abs(lines - actuals))
    weighted_avg_mae = np.mean(np.abs(weighted_avg - actuals))
    career_avg_mae = np.mean(np.abs(career_avg - actuals))
    mean_baseline_mae = np.mean(np.abs(np.mean(actuals) - actuals))

    print(f"Model MAE:              {model_mae:.2f} yards")
    print(f"Betting Line MAE:       {line_mae:.2f} yards")
    print(f"weighted_avg MAE:       {weighted_avg_mae:.2f} yards")
    print(f"career_avg MAE:         {career_avg_mae:.2f} yards")
    print(f"Mean baseline MAE:      {mean_baseline_mae:.2f} yards")
    print()

    print(f"Model vs Line:          {model_mae - line_mae:+.2f} yards ({100*(model_mae - line_mae)/line_mae:+.1f}%)")
    print(f"Model vs weighted_avg:  {model_mae - weighted_avg_mae:+.2f} yards ({100*(model_mae - weighted_avg_mae)/weighted_avg_mae:+.1f}%)")
    print()

    if line_mae < model_mae:
        print(f"DIAGNOSIS: Betting lines are {line_mae - model_mae:.2f} yards BETTER than our model")
        print("           Sportsbooks have superior predictions.")
    else:
        print(f"DIAGNOSIS: Our model is {model_mae - line_mae:.2f} yards BETTER than betting lines!")
        print("           (But still losing money on binary over/under)")
    print()

    if weighted_avg_mae <= model_mae + 0.5:
        print(f"DIAGNOSIS: Model barely improves over just using weighted_avg")
        print(f"           ({model_mae:.2f} vs {weighted_avg_mae:.2f})")
        print("           All 41 other features add minimal value.")
    print()

    # ========================================================================
    # TEST 4: FEATURE REDUNDANCY
    # ========================================================================
    print("="*80)
    print("TEST 4: FEATURE REDUNDANCY - Does weighted_avg dominate?")
    print("="*80)
    print()

    # Correlation between weighted_avg and other top features
    feature_correlations = {}
    for feature_name in ['career_avg', 'targets_season_avg', 'is_dome', 'is_home',
                         'vegas_total', 'targets_3wk_avg']:
        try:
            feature_vals = matched_df.select(feature_name).to_numpy().flatten()
            # Only calculate for non-NaN values
            valid_mask = ~(np.isnan(weighted_avg) | np.isnan(feature_vals))
            if np.sum(valid_mask) > 10:
                corr = np.corrcoef(weighted_avg[valid_mask], feature_vals[valid_mask])[0, 1]
                feature_correlations[feature_name] = corr
        except:
            pass

    print("Correlation between weighted_avg and other top features:")
    for feat, corr in sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:25s}: {corr:+.4f}")
    print()

    high_corr = [f for f, c in feature_correlations.items() if abs(c) > 0.7]
    if high_corr:
        print(f"DIAGNOSIS: {len(high_corr)} features have >0.7 correlation with weighted_avg:")
        for f in high_corr:
            print(f"           - {f}")
        print("           These features are likely redundant.")
    else:
        print("DIAGNOSIS: Most features provide independent signal from weighted_avg")
    print()

    # ========================================================================
    # TEST 5: WORST PREDICTIONS
    # ========================================================================
    print("="*80)
    print("TEST 5: WORST PREDICTIONS - What specific bets fail?")
    print("="*80)
    print()

    # Binary prediction accuracy
    ml_over = predictions > lines
    actual_over = actuals > lines
    correct = ml_over == actual_over
    accuracy = np.mean(correct)

    print(f"Overall binary accuracy: {100*accuracy:.1f}% ({np.sum(correct)}/{len(correct)})")
    print()

    # Find worst predictions
    worst_indices = np.argsort(abs_errors)[-20:][::-1]

    print("Top 20 Worst Predictions (by absolute error):")
    print(f"{'Pred':>6s} {'Line':>6s} {'Actual':>6s} {'Error':>7s} {'Result':>10s} {'Vol':>5s} {'Home':>4s} {'Dome':>4s}")
    print("-" * 65)

    for i, idx in enumerate(worst_indices):
        pred = predictions[idx]
        line = lines[idx]
        actual = actuals[idx]
        error = errors[idx]
        vol = targets_season_avg[idx]
        home = "Y" if is_home[idx] == 1 else "N"
        dome = "Y" if is_dome[idx] == 1 else "N"

        if ml_over[idx] and not actual_over[idx]:
            result = "Over LOSS"
        elif not ml_over[idx] and actual_over[idx]:
            result = "Under LOSS"
        else:
            result = "WIN"

        print(f"{pred:6.1f} {line:6.1f} {actual:6.1f} {error:+7.1f} {result:>10s} {vol:5.1f} {home:>4s} {dome:>4s}")
    print()

    # Systematic failures
    bad_over = (ml_over) & (~actual_over) & (abs_errors > 20)
    bad_under = (~ml_over) & (actual_over) & (abs_errors > 20)

    print(f"Systematic Over failures (predicted Over, actual Under by 20+ yards): {np.sum(bad_over)}")
    print(f"Systematic Under failures (predicted Under, actual Over by 20+ yards): {np.sum(bad_under)}")
    print()

    if np.sum(bad_over) > 0:
        print("Patterns in bad Over predictions:")
        print(f"  Home games: {100*np.mean(is_home[bad_over]):.1f}%")
        print(f"  Dome games: {100*np.mean(is_dome[bad_over]):.1f}%")
        print(f"  Avg volume: {np.mean(targets_season_avg[bad_over]):.2f} targets/game")
    print()

    # ========================================================================
    # TEST 6: INFORMATION GAP
    # ========================================================================
    print("="*80)
    print("TEST 6: INFORMATION GAP - What don't we have?")
    print("="*80)
    print()

    # Get feature importance
    importance = model.get_feature_importance()

    # Categorize features
    info_features = {
        'injury_status_score': importance.get('injury_status_score', 0),
        'extreme_weather': importance.get('extreme_weather', 0),
        'game_wind': importance.get('game_wind', 0),
        'game_temp': importance.get('game_temp', 0),
    }

    print("Importance of 'information advantage' features:")
    for feat, imp in sorted(info_features.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:25s}: {100*imp:5.2f}%")
    print()

    total_info_importance = sum(info_features.values())
    print(f"Total importance from injury/weather: {100*total_info_importance:.2f}%")
    print()

    if total_info_importance < 0.05:
        print("DIAGNOSIS: Injury and weather features contribute <5% importance")
        print("           These 'information advantage' features aren't helping.")
    print()

    print("What sportsbooks likely have that we DON'T:")
    print("  - Real-time injury reports (practice status, snap count trends)")
    print("  - CB matchup quality (our opponent has which specific CB?)")
    print("  - Coaching tendencies (game script, target distribution in specific matchups)")
    print("  - Sharp money movement (line movement indicates where pros are betting)")
    print("  - Proprietary models calibrated on years of data")
    print("  - Better weather data (real-time wind readings at kickoff)")
    print()

    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================
    print("="*80)
    print("EXECUTIVE SUMMARY - ROOT CAUSE DIAGNOSIS")
    print("="*80)
    print()

    issues = []

    if corr_pred_line > 0.8:
        issues.append(f"Model heavily influenced by betting lines (r={corr_pred_line:.3f})")

    if line_mae < model_mae:
        issues.append(f"Betting lines are {line_mae - model_mae:.1f} yards more accurate")

    if weighted_avg_mae - model_mae < 1.0:
        issues.append(f"41 features barely improve over weighted_avg ({weighted_avg_mae - model_mae:.1f} yards)")

    if abs(mean_error) > 2:
        issues.append(f"Systematic {'over' if mean_error > 0 else 'under'}-prediction bias ({mean_error:+.1f} yards)")

    if accuracy < 0.45:
        issues.append(f"Binary accuracy critically low ({100*accuracy:.1f}% vs 52.4% breakeven)")

    print("PRIMARY ISSUES IDENTIFIED:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    print()

    print("RECOMMENDATIONS:")

    if line_mae < model_mae - 2:
        print("1. ACCEPT DEFEAT: Sportsbooks have better information and models")
        print("   Consider focusing on props with less efficient markets")

    if weighted_avg_mae - model_mae < 1.0:
        print("2. FEATURE PROBLEM: Current features don't add value beyond recent performance")
        print("   Need genuinely NEW information (CB matchups, route data, etc.)")

    if accuracy < 0.45:
        print("3. CALIBRATION PROBLEM: Even if yards are close, binary predictions are wrong")
        print("   Try classification model (predict Over/Under directly)")
        print("   Or use probability calibration on regression outputs")

    if corr_pred_line > 0.8:
        print("4. CIRCULARITY: Model is influenced by the lines we're trying to beat")
        print("   Remove vegas_total/vegas_spread features and retrain")

    print()
    print("="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
