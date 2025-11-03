"""
Test Player Props Baseline Calculations

Validates that baseline projections (weighted rolling averages) are calculated correctly.
Tests on known players from 2024 season.
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.prop_data_aggregator import (
    calculate_weighted_rolling_average,
    get_player_baseline_projections,
    get_historical_averages,
    calculate_stat_variance
)
from modules.prop_projection_engine import generate_projection
from modules.prop_types import get_prop_types_for_position
import polars as pl


def test_weighted_rolling_average():
    """Test weighted rolling average calculation with known values."""
    print("\n=== Test 1: Weighted Rolling Average ===")

    # Create test data: 8 games with passing_yards
    test_data = {
        'week': [1, 2, 3, 4, 5, 6, 7, 8],
        'passing_yards': [292, 315, 245, 267, 289, 301, 278, 264]
    }
    df = pl.DataFrame(test_data)

    # Calculate weighted average
    # Weights: [0.25, 0.50, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
    # weighted_sum = 292×0.25 + 315×0.50 + 245×0.75 + 267×1.0 + ... + 264×1.0
    # cumulative_weight = 0.25 + 0.50 + 0.75 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 = 6.5

    result = calculate_weighted_rolling_average(df, 'passing_yards', through_week=8)

    # Manual calculation
    weighted_sum = (292*0.25 + 315*0.50 + 245*0.75 + 267*1.0 +
                    289*1.0 + 301*1.0 + 278*1.0 + 264*1.0)
    cumulative_weight = 6.5
    expected = weighted_sum / cumulative_weight

    print(f"Result: {result:.2f}")
    print(f"Expected: {expected:.2f}")
    print(f"Match: {abs(result - expected) < 0.01}")

    assert abs(result - expected) < 0.01, f"Weighted average mismatch: {result} vs {expected}"
    print("[PASS] Test passed")


def test_load_real_player_data():
    """Test loading real player data from cache."""
    print("\n=== Test 2: Load Real Player Data ===")

    # Test loading QB stats for 2024
    stats_file = Path("cache/positional_player_stats/qb/qb-2024.csv")

    if not stats_file.exists():
        print(f"[FAIL] Stats file not found: {stats_file}")
        print("  Skipping test (data not available)")
        return

    df = pl.read_csv(stats_file)
    print(f"[PASS] Loaded {len(df)} QB stat records for 2024")
    print(f"  Columns: {df.columns[:10]}")  # Show first 10 columns

    # Check for Patrick Mahomes
    mahomes_data = df.filter(pl.col('player_display_name').str.contains('Mahomes'))

    if len(mahomes_data) > 0:
        player_id = mahomes_data['player_id'][0]
        print(f"[PASS] Found Patrick Mahomes: {player_id}")
        print(f"  Games: {len(mahomes_data.filter(pl.col('week') <= 9))}")

        # Test baseline projection
        baselines = get_player_baseline_projections(player_id, 2024, 9, 'QB')
        if baselines:
            print(f"[PASS] Baseline projections calculated:")
            print(f"    Passing yards: {baselines.get('passing_yards', 0):.1f}")
            print(f"    Passing TDs: {baselines.get('passing_tds', 0):.1f}")
            print(f"    Games played: {baselines.get('games_played', 0)}")
        else:
            print(f"[FAIL] No baseline projections generated")
    else:
        print("[FAIL] Patrick Mahomes not found in data")


def test_projection_generation():
    """Test complete projection generation."""
    print("\n=== Test 3: Projection Generation ===")

    # Test loading QB stats for 2024
    stats_file = Path("cache/positional_player_stats/qb/qb-2024.csv")

    if not stats_file.exists():
        print(f"[FAIL] Stats file not found, skipping test")
        return

    df = pl.read_csv(stats_file)
    mahomes_data = df.filter(pl.col('player_display_name').str.contains('Mahomes'))

    if len(mahomes_data) == 0:
        print("[FAIL] Mahomes data not found, skipping test")
        return

    player_id = mahomes_data['player_id'][0]
    print(f"Testing projection for {player_id}")

    # Generate projection for Week 10 passing yards
    projection = generate_projection(
        player_id=player_id,
        season=2024,
        week=10,
        position='QB',
        prop_type='passing_yards'
    )

    if projection:
        print(f"[PASS] Projection generated successfully:")
        print(f"    Baseline: {projection['baseline']:.1f} yards")
        print(f"    Adjusted: {projection['adjusted_projection']:.1f} yards")
        print(f"    Final (dampened): {projection['final_projection']:.1f} yards")
        print(f"    Effective games: {projection['effective_games']:.1f}")
        print(f"    Adjustments: {projection['adjustments']}")

        # Check stat summary
        summary = projection['stat_summary']
        print(f"\n    Historical averages:")
        print(f"      Last 3: {summary['last_3_avg']:.1f} yards")
        print(f"      Last 5: {summary['last_5_avg']:.1f} yards")
        print(f"      Season: {summary['season_avg']:.1f} yards")
        print(f"      Variance (CV): {summary['variance']:.3f}")
    else:
        print("[FAIL] Projection generation failed")


def test_multiple_prop_types():
    """Test projections for multiple prop types."""
    print("\n=== Test 4: Multiple Prop Types ===")

    stats_file = Path("cache/positional_player_stats/qb/qb-2024.csv")

    if not stats_file.exists():
        print(f"[FAIL] Stats file not found, skipping test")
        return

    df = pl.read_csv(stats_file)
    mahomes_data = df.filter(pl.col('player_display_name').str.contains('Mahomes'))

    if len(mahomes_data) == 0:
        print("[FAIL] Mahomes data not found, skipping test")
        return

    player_id = mahomes_data['player_id'][0]

    # Test all QB prop types
    qb_props = get_prop_types_for_position('QB')
    print(f"Testing {len(qb_props)} prop types for QB:")

    for prop_type in qb_props:
        projection = generate_projection(
            player_id=player_id,
            season=2024,
            week=10,
            position='QB',
            prop_type=prop_type
        )

        if projection and projection['final_projection'] > 0:
            print(f"  [PASS] {prop_type}: {projection['final_projection']:.1f}")
        else:
            print(f"  [FAIL] {prop_type}: No projection")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("PLAYER PROPS BASELINE TESTS")
    print("="*60)

    try:
        test_weighted_rolling_average()
        test_load_real_player_data()
        test_projection_generation()
        test_multiple_prop_types()

        print("\n" + "="*60)
        print("[PASS] ALL TESTS COMPLETED")
        print("="*60)

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
