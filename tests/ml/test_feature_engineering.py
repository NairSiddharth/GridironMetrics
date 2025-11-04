"""
Test ML Feature Engineering

Validates that feature extraction works correctly and produces features
in expected ranges without data leakage.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.ml_feature_engineering import PropFeatureEngineer


def test_feature_extraction_qb_passing_yards():
    """Test feature engineering for QB passing yards prop."""
    engineer = PropFeatureEngineer()

    # Patrick Mahomes 2024 Week 10
    features = engineer.engineer_features(
        player_id="00-0033873",
        season=2024,
        week=10,
        position='QB',
        prop_type='passing_yards',
        opponent_team='BUF'
    )

    # Validate feature presence
    required_features = [
        'weighted_avg', 'last_3_avg', 'last_5_avg', 'career_avg',
        'variance_cv', 'games_played', 'opp_def_pass_ypa',
        'success_rate_3wk', 'opponent', 'position'
    ]
    for feature in required_features:
        assert feature in features, f"Missing feature: {feature}"

    # Validate feature ranges for QB
    assert 150 <= features['weighted_avg'] <= 450, f"Weighted avg {features['weighted_avg']} out of range"
    assert 0 <= features['success_rate_3wk'] <= 1, "Success rate out of range"
    assert features['opp_def_pass_ypa'] > 0, "Opponent defense should be positive"
    assert features['games_played'] >= 1, "Games played should be >= 1"
    assert 0 <= features['variance_cv'] <= 2, "Variance CV should be reasonable"

    # Validate categorical features
    assert features['opponent'] == 'BUF'
    assert features['position'] == 'QB'
    assert features['week'] == 10
    assert features['season'] == 2024

    print(f"\n[PASS] QB passing yards: {len(features)} features extracted successfully")


def test_feature_extraction_handles_missing_data():
    """Test that feature engineering handles missing data gracefully."""
    engineer = PropFeatureEngineer()

    # Fictional player with no data
    features = engineer.engineer_features(
        player_id="00-0000000",  # Doesn't exist
        season=2024,
        week=5,
        position='QB',
        prop_type='passing_yards',
        opponent_team='KC'
    )

    # Should return defaults, not crash
    assert 'weighted_avg' in features
    assert features['weighted_avg'] == 0.0  # No data
    assert features['career_avg'] >= 0  # May be 0 if no career data
    assert features['games_played'] == 0

    print(f"\n[PASS] Missing data handled gracefully")


def test_feature_extraction_no_future_data():
    """Test that features only use data through week-1."""
    engineer = PropFeatureEngineer()

    # Week 9 features should only use weeks 1-8 data
    features_week9 = engineer.engineer_features(
        player_id="00-0033873",
        season=2024,
        week=9,
        position='QB',
        prop_type='passing_yards',
        opponent_team='TB'
    )

    # Week 10 features should only use weeks 1-9 data
    features_week10 = engineer.engineer_features(
        player_id="00-0033873",
        season=2024,
        week=10,
        position='QB',
        prop_type='passing_yards',
        opponent_team='BUF'
    )

    # Week 10 should have more games played than week 9
    assert features_week10['games_played'] > features_week9['games_played'], \
        "Week 10 should have more games than week 9"

    # Weighted avg should differ (more data in week 10)
    assert features_week10['weighted_avg'] != features_week9['weighted_avg'], \
        "Features should differ between weeks"

    print(f"\n[PASS] No future data leakage detected")


def test_feature_types():
    """Test that feature types are correct."""
    engineer = PropFeatureEngineer()

    features = engineer.engineer_features(
        player_id="00-0033873",
        season=2024,
        week=10,
        position='QB',
        prop_type='passing_yards',
        opponent_team='BUF'
    )

    # Numeric features should be float or int
    numeric_features = [
        'weighted_avg', 'last_3_avg', 'career_avg', 'variance_cv',
        'opp_def_pass_ypa', 'success_rate_3wk', 'game_temp'
    ]
    for feature in numeric_features:
        assert isinstance(features[feature], (int, float)), \
            f"{feature} should be numeric, got {type(features[feature])}"

    # Categorical features should be string or int
    categorical_features = ['opponent', 'position']
    for feature in categorical_features:
        assert isinstance(features[feature], (str, int)), \
            f"{feature} should be string or int, got {type(features[feature])}"

    print(f"\n[PASS] Feature types validated")


def test_position_specific_features():
    """Test that position-specific features are generated correctly."""
    engineer = PropFeatureEngineer()

    # RB should have blocking quality features
    rb_features = engineer.engineer_features(
        player_id="00-0036945",  # Example RB
        season=2024,
        week=10,
        position='RB',
        prop_type='rushing_yards',
        opponent_team='KC'
    )

    # Check for RB-specific features
    if rb_features['games_played'] > 0:
        assert 'player_ypc' in rb_features
        assert 'team_ypc' in rb_features
        assert 'ypc_diff_pct' in rb_features

    print(f"\n[PASS] Position-specific features validated")


if __name__ == "__main__":
    # Run tests
    print("="*60)
    print("ML Feature Engineering Tests")
    print("="*60)

    try:
        test_feature_extraction_qb_passing_yards()
        test_feature_extraction_handles_missing_data()
        test_feature_extraction_no_future_data()
        test_feature_types()
        test_position_specific_features()

        print("\n" + "="*60)
        print("[SUCCESS] All tests passed")
        print("="*60)

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
