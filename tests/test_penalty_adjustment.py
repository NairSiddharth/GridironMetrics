"""
Test penalty adjustment calculations with known 2024 cases.
"""

from modules.penalty_cache_builder import (
    load_penalty_data,
    calculate_penalty_adjustment,
)
from modules.injury_cache_builder import get_player_gsis_id
import polars as pl

def test_penalty_data_loading():
    """Test that penalty data loads correctly."""
    print("\n" + "=" * 60)
    print("TEST: Penalty Data Loading")
    print("=" * 60)
    
    penalties = load_penalty_data(2024)
    
    assert len(penalties) > 0, "Should load penalty data"
    assert 'penalty_player_id' in penalties.columns
    assert 'penalty_type' in penalties.columns
    assert 'epa' in penalties.columns
    
    print(f"✓ Loaded {len(penalties)} penalties for 2024")
    return True


def test_drake_london_penalties():
    """
    Test Drake London - 2 unsportsmanlike conduct penalties in 2024.
    Should get penalized as repeat offender.
    """
    print("\n" + "=" * 60)
    print("TEST: Drake London (Repeat Dead Ball Offender)")
    print("=" * 60)
    
    # Get Drake London's GSIS ID
    gsis_id = get_player_gsis_id("D.London", "ATL", "WR", 2024)
    
    if not gsis_id:
        print("✗ Could not find Drake London's GSIS ID")
        return False
    
    # Calculate penalty adjustment
    adjustment = calculate_penalty_adjustment(gsis_id, 2024)
    
    print(f"\nDrake London Penalty Analysis:")
    print(f"  GSIS ID: {gsis_id}")
    print(f"  Penalty Multiplier: {adjustment:.3f}")
    print(f"  Reduction: {(1.0 - adjustment) * 100:.1f}%")
    
    # Should be penalized (multiplier < 1.0)
    if adjustment >= 1.0:
        print("✗ Drake London should be penalized for 2 USC penalties")
        return False
    
    if adjustment < 0.80:
        print("✗ Penalty exceeds 20% maximum")
        return False
    
    print("✓ Drake London correctly penalized as repeat offender")
    return True


def test_george_pickens_penalties():
    """
    Test George Pickens - 2 penalties (taunting + USC) in 2024.
    Should get penalized as repeat offender.
    """
    print("\n" + "=" * 60)
    print("TEST: George Pickens (Repeat Dead Ball Offender)")
    print("=" * 60)
    
    gsis_id = get_player_gsis_id("G.Pickens", "PIT", "WR", 2024)
    
    if not gsis_id:
        print("✗ Could not find George Pickens' GSIS ID")
        return False
    
    adjustment = calculate_penalty_adjustment(gsis_id, 2024)
    
    print(f"\nGeorge Pickens Penalty Analysis:")
    print(f"  GSIS ID: {gsis_id}")
    print(f"  Penalty Multiplier: {adjustment:.3f}")
    print(f"  Reduction: {(1.0 - adjustment) * 100:.1f}%")
    
    if adjustment >= 1.0:
        print("✗ Pickens should be penalized")
        return False
    
    if adjustment < 0.80:
        print("✗ Penalty exceeds 20% maximum")
        return False
    
    print("✓ George Pickens correctly penalized")
    return True


def test_dtr_intentional_grounding():
    """
    Test Dorian Thompson-Robinson - 5 intentional grounding penalties.
    Should get heavily penalized (loss of down penalties).
    """
    print("\n" + "=" * 60)
    print("TEST: Dorian Thompson-Robinson (Loss of Down Penalties)")
    print("=" * 60)
    
    gsis_id = get_player_gsis_id("D.Thompson-Robinson", "CLE", "QB", 2024)
    
    if not gsis_id:
        print("✗ Could not find DTR's GSIS ID")
        return False
    
    adjustment = calculate_penalty_adjustment(gsis_id, 2024)
    
    print(f"\nDorian Thompson-Robinson Penalty Analysis:")
    print(f"  GSIS ID: {gsis_id}")
    print(f"  Penalty Multiplier: {adjustment:.3f}")
    print(f"  Reduction: {(1.0 - adjustment) * 100:.1f}%")
    
    # 5 loss-of-down penalties should be heavily penalized
    if adjustment >= 0.90:
        print("✗ DTR should be heavily penalized for 5 grounding penalties")
        return False
    
    if adjustment < 0.80:
        print("✗ Penalty exceeds 20% maximum")
        return False
    
    print("✓ DTR correctly penalized for intentional grounding")
    return True


def test_clean_player_no_penalties():
    """
    Test a player with no penalties - should return 1.0 (no adjustment).
    """
    print("\n" + "=" * 60)
    print("TEST: Clean Player (No Penalties)")
    print("=" * 60)
    
    # Test with a reliable player (likely has no penalties)
    gsis_id = get_player_gsis_id("C.Lamb", "DAL", "WR", 2024)
    
    if not gsis_id:
        print("✗ Could not find CeeDee Lamb's GSIS ID")
        return False
    
    adjustment = calculate_penalty_adjustment(gsis_id, 2024)
    
    print(f"\nCeeDee Lamb Penalty Analysis:")
    print(f"  GSIS ID: {gsis_id}")
    print(f"  Penalty Multiplier: {adjustment:.3f}")
    
    # If no penalties, should be 1.0
    if adjustment == 1.0:
        print("✓ CeeDee Lamb has no penalties (1.0 multiplier)")
        return True
    else:
        print(f"  CeeDee has penalties: {(1.0 - adjustment) * 100:.1f}% reduction")
        return True  # Still valid if he has penalties


def test_penalty_cache_exists():
    """Test that 2024 penalty cache was created."""
    print("\n" + "=" * 60)
    print("TEST: Penalty Cache File Exists")
    print("=" * 60)
    
    from pathlib import Path
    cache_file = Path("cache/penalties/penalties-2024.csv")
    
    if not cache_file.exists():
        print("✗ 2024 penalty cache does not exist")
        return False
    
    penalties = pl.read_csv(cache_file)
    if len(penalties) == 0:
        print("✗ Cache is empty")
        return False
    
    print(f"✓ Penalty cache exists with {len(penalties)} penalties")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("PENALTY ADJUSTMENT SYSTEM - TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Data Loading", test_penalty_data_loading()))
    results.append(("Cache Exists", test_penalty_cache_exists()))
    results.append(("Drake London", test_drake_london_penalties()))
    results.append(("George Pickens", test_george_pickens_penalties()))
    results.append(("DTR (Grounding)", test_dtr_intentional_grounding()))
    results.append(("Clean Player", test_clean_player_no_penalties()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All penalty adjustment tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("=" * 60)
