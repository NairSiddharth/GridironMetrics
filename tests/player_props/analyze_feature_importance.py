"""
Analyze Feature Importance for WR Receiving Yards Models

Extracts and compares feature importance from both full and high-volume models
to understand which features are contributing most to predictions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.ml_ensemble import PropEnsembleModel
from modules.constants import CACHE_DIR

def main():
    print("="*80)
    print("WR RECEIVING YARDS - FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    # Load both models
    print("\nLoading models...")
    full_model = PropEnsembleModel(prop_type='receiving_yards_wr')
    full_model.load(str(Path(CACHE_DIR) / "ml_models" / "receiving_yards_wr_ensemble.pkl"))
    print("  Full model loaded")

    high_vol_model = PropEnsembleModel(prop_type='receiving_yards_wr')
    high_vol_model.load(str(Path(CACHE_DIR) / "ml_models" / "receiving_yards_wr_ensemble_high_volume.pkl"))
    print("  High-volume model loaded")

    # Extract feature importance from both models
    print("\n" + "="*80)
    print("FULL MODEL FEATURE IMPORTANCE (All WRs)")
    print("="*80)
    full_model.print_feature_importance(top_n=45)

    print("\n" + "="*80)
    print("HIGH-VOLUME MODEL FEATURE IMPORTANCE (>=4.5 targets/game)")
    print("="*80)
    high_vol_model.print_feature_importance(top_n=45)

    # Get feature importance dictionaries for comparison
    full_importance = full_model.get_feature_importance()
    hv_importance = high_vol_model.get_feature_importance()

    # Compare the two models
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE COMPARISON")
    print("="*80)
    print(f"{'Feature':<40} {'Full Rank':<12} {'HV Rank':<12} {'Rank Diff':<12}")
    print("-"*80)

    # Get rankings
    full_ranked = sorted(full_importance.items(), key=lambda x: x[1], reverse=True)
    hv_ranked = sorted(hv_importance.items(), key=lambda x: x[1], reverse=True)

    full_rank_dict = {feat: idx+1 for idx, (feat, _) in enumerate(full_ranked)}
    hv_rank_dict = {feat: idx+1 for idx, (feat, _) in enumerate(hv_ranked)}

    # Show top 20 from full model with comparison
    for idx, (feature, importance) in enumerate(full_ranked[:20], 1):
        hv_rank = hv_rank_dict.get(feature, "N/A")
        if hv_rank != "N/A":
            rank_diff = hv_rank - idx
            diff_str = f"{rank_diff:+d}" if rank_diff != 0 else "0"
        else:
            diff_str = "N/A"

        print(f"{feature:<40} {idx:<12} {str(hv_rank):<12} {diff_str:<12}")

    # Identify features that improved/declined significantly
    print("\n" + "="*80)
    print("SIGNIFICANT RANKING CHANGES (>10 positions)")
    print("="*80)

    big_movers = []
    for feature in full_importance:
        if feature in hv_rank_dict:
            full_rank = full_rank_dict[feature]
            hv_rank = hv_rank_dict[feature]
            diff = hv_rank - full_rank
            if abs(diff) > 10:
                big_movers.append((feature, full_rank, hv_rank, diff))

    big_movers.sort(key=lambda x: abs(x[3]), reverse=True)

    if big_movers:
        print(f"{'Feature':<40} {'Full Rank':<12} {'HV Rank':<12} {'Change':<12}")
        print("-"*80)
        for feature, full_rank, hv_rank, diff in big_movers:
            direction = "UP" if diff < 0 else "DOWN"
            print(f"{feature:<40} {full_rank:<12} {hv_rank:<12} {diff:+d} ({direction})")
    else:
        print("No significant ranking changes found.")

    # Identify bottom features (potential noise)
    print("\n" + "="*80)
    print("BOTTOM 10 FEATURES (Potential Noise)")
    print("="*80)
    print(f"{'Feature':<40} {'Full Importance':<20} {'HV Importance':<20}")
    print("-"*80)

    for idx in range(-10, 0):
        feature, importance = full_ranked[idx]
        hv_imp = hv_importance.get(feature, 0.0)
        print(f"{feature:<40} {importance:<20.6f} {hv_imp:<20.6f}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Insights:")
    print("1. Check if new features (Phase 1-5) are in top 20")
    print("2. Identify low-importance features to potentially remove")
    print("3. Look for features that rank very differently between models")


if __name__ == "__main__":
    main()
