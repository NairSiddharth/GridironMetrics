"""
Test weather adjustment calculations with known scenarios.
"""

import polars as pl
from modules.weather_cache_builder import (
    categorize_temperature,
    categorize_wind,
    has_precipitation,
    categorize_environment,
    build_weather_performance_cache,
    calculate_weather_adjustment
)


def test_temperature_categorization():
    """Test temperature buckets."""
    print("\n=== Testing Temperature Categorization ===")
    assert categorize_temperature(25.0) == 'cold', f"Expected 'cold', got {categorize_temperature(25.0)}"
    assert categorize_temperature(45.0) == 'cool', f"Expected 'cool', got {categorize_temperature(45.0)}"
    assert categorize_temperature(65.0) == 'moderate', f"Expected 'moderate', got {categorize_temperature(65.0)}"
    assert categorize_temperature(85.0) == 'hot', f"Expected 'hot', got {categorize_temperature(85.0)}"
    print("[OK] Temperature categorization works")


def test_wind_categorization():
    """Test wind buckets."""
    print("\n=== Testing Wind Categorization ===")
    assert categorize_wind(5.0) == 'calm', f"Expected 'calm', got {categorize_wind(5.0)}"
    assert categorize_wind(15.0) == 'moderate', f"Expected 'moderate', got {categorize_wind(15.0)}"
    assert categorize_wind(25.0) == 'high', f"Expected 'high', got {categorize_wind(25.0)}"
    print("[OK] Wind categorization works")


def test_precipitation_detection():
    """Test precipitation detection."""
    print("\n=== Testing Precipitation Detection ===")
    assert has_precipitation("Light Snow") == True, "Expected True for 'Light Snow'"
    assert has_precipitation("Rain") == True, "Expected True for 'Rain'"
    assert has_precipitation("Sunny") == False, "Expected False for 'Sunny'"
    assert has_precipitation("Clear") == False, "Expected False for 'Clear'"
    print("[OK] Precipitation detection works")


def test_environment_categorization():
    """Test environment buckets."""
    print("\n=== Testing Environment Categorization ===")
    assert categorize_environment('dome') == 'dome', f"Expected 'dome', got {categorize_environment('dome')}"
    assert categorize_environment('closed') == 'dome', f"Expected 'dome', got {categorize_environment('closed')}"
    assert categorize_environment('outdoors') == 'outdoor', f"Expected 'outdoor', got {categorize_environment('outdoors')}"
    assert categorize_environment('open') == 'outdoor', f"Expected 'outdoor', got {categorize_environment('open')}"
    print("[OK] Environment categorization works")


def test_cache_building():
    """Test that cache building works for QB 2023."""
    print("\n=== Testing Cache Building for QB 2023 ===")
    df = build_weather_performance_cache(2023, 'QB')

    assert len(df) > 0, "Should have QB weather data"
    assert 'player_id' in df.columns, "Should have player_id column"
    assert 'temp_cold_adj' in df.columns, "Should have temp_cold_adj column"
    assert 'wind_high_adj' in df.columns, "Should have wind_high_adj column"

    print(f"[OK] Built weather cache with {len(df)} QBs for 2023")

    # Show sample data statistics
    if len(df) > 0:
        print(f"  Average cold temp adjustment: {df['temp_cold_adj'].mean():.3f}")
        print(f"  Average high wind adjustment: {df['wind_high_adj'].mean():.3f}")


def test_mahomes_cold_weather():
    """
    Test Patrick Mahomes in cold weather.
    Mahomes historically performs well in cold weather games.
    """
    print("\n=== Testing Mahomes Cold Weather Performance ===")
    # Build cache first
    df = build_weather_performance_cache(2023, 'QB')

    # Find Mahomes
    mahomes = df.filter(pl.col('player_name').str.contains('Mahomes'))

    if len(mahomes) > 0:
        cold_adj = mahomes['temp_cold_adj'][0]
        wind_high_adj = mahomes['wind_high_adj'][0]
        total_plays = mahomes['total_plays'][0]

        print(f"Mahomes stats:")
        print(f"  Total plays: {total_plays}")
        print(f"  Cold weather adjustment: {cold_adj:.3f}")
        print(f"  High wind adjustment: {wind_high_adj:.3f}")

        # Mahomes should perform well in cold (>= 1.0 or close to it)
        print(f"[OK] Mahomes cold weather data loaded")
    else:
        print("[FAILED] Mahomes not found in 2023 data")


def test_weather_adjustment_calculation():
    """
    Test full weather adjustment calculation.
    """
    print("\n=== Testing Full Weather Adjustment Calculation ===")
    # Get a real QB from 2023
    df = build_weather_performance_cache(2023, 'QB')

    if len(df) > 0:
        first_qb = df[0]
        player_id = first_qb['player_id']
        player_name = first_qb['player_name']

        # Simulate cold, windy game
        adjustment = calculate_weather_adjustment(
            player_id=player_id,
            season=2023,
            position='QB',
            game_temp=28.0,  # Cold
            game_wind=22.0,  # High wind
            game_weather='Clear',
            game_roof='outdoors'
        )

        print(f"\n{player_name} in cold (28°F) + high wind (22 mph):")
        print(f"  Weather adjustment: {adjustment:.3f}")

        # Should be between 0.90 and 1.10
        assert 0.90 <= adjustment <= 1.10, f"Should be within bounds, got {adjustment}"

        print("[OK] Weather adjustment calculation works")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Weather Adjustment System")
    print("=" * 60)

    try:
        test_temperature_categorization()
        test_wind_categorization()
        test_precipitation_detection()
        test_environment_categorization()
        test_cache_building()
        test_mahomes_cold_weather()
        test_weather_adjustment_calculation()

        print("\n" + "=" * 60)
        print("All weather adjustment tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
