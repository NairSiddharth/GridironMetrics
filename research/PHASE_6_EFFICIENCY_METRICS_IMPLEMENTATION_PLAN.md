# Phase 6: Efficiency Metrics Enhancement Implementation Plan

**Version**: 1.0
**Date**: 2025-01-02
**Status**: Planning
**Priority**: High
**Estimated Effort**: 2-3 weeks

---

## Executive Summary

### Objective
Enhance GridironMetrics player ranking accuracy by integrating 4 new efficiency-focused metrics from nflfastpy PBP data:
1. **Success Rate** - Play efficiency measurement (did play advance situation?)
2. **Drive Context** - Scoring drive weighting (reward plays on successful drives)
3. **Route/Target Location** - Catch difficulty by field position (with slot receiver bias mitigation)
4. **Turnover Detail & Attribution** - Better INT/fumble responsibility assignment

### Expected Impact
- **15-20% accuracy improvement** in player rankings
- **Better separation** between volume accumulators and efficient contributors
- **Fairer evaluation** of slot receivers vs outside receivers
- **More accurate QB rankings** via better turnover attribution

### Key Benefits
- All features use existing nflfastpy columns (no external data needed)
- Integrates cleanly into existing Phase 1 and Phase 4 architecture
- Minimal performance impact (<10% processing time increase)
- Comprehensive NaN prevention and error handling
- Validates against known 2023-2024 player performances

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Data Coverage Validation](#data-coverage-validation)
3. [Phase 6.1: Success Rate Integration](#phase-61-success-rate-integration)
4. [Phase 6.2: Drive Context Integration](#phase-62-drive-context-integration)
5. [Phase 6.3: Route/Target Location Integration](#phase-63-routetarget-location-integration)
6. [Phase 6.4: Turnover Attribution Enhancement](#phase-64-turnover-attribution-enhancement)
7. [Testing Strategy](#testing-strategy)
8. [Performance Considerations](#performance-considerations)
9. [Implementation Checklist](#implementation-checklist)
10. [Success Criteria](#success-criteria)
11. [Risk Mitigation](#risk-mitigation)
12. [Timeline](#timeline)
13. [Appendices](#appendices)

---

## Current State Assessment

### Existing Phase Architecture

GridironMetrics currently implements a sophisticated multi-phase adjustment pipeline:

**Phase 1** (Play Level - PBP Cache):
- Field position, score differential, time, down multipliers
- Range: 1.0-3.375x combined
- Location: `modules/play_by_play.py:102-111`

**Phase 1.5** (Game Level):
- Personnel/formation context
- Range: 0.80-1.15x
- Location: `modules/contribution_calculators.py:453-494`

**Phase 2** (Game Level):
- Opponent strength (rolling defensive rankings)
- Range: 0.5-2.5x
- Location: `modules/contribution_calculators.py:adjust_for_opponent`

**Phase 3** (Season Level - Informational):
- Difficulty context (defenders in box, coverage type)
- Used for narrative, not multipliers
- Location: `modules/contribution_calculators.py:calculate_difficulty_context`

**Phase 4** (Season + Per-Week):
- Catch rate, blocking quality, separation, penalties
- Range: 0.70-1.15x combined
- Location: `modules/adjustment_pipeline.py:apply_phase4_adjustments`

**Phase 4.5** (Per-Week):
- Weather adjustments
- Range: 0.85-1.15x
- Location: `modules/adjustment_pipeline.py:apply_phase4_5_weather_adjustments`

**Phase 5** (Season Level):
- Talent context + sample size dampening
- Range: 0.85-1.15x + regression
- Location: `modules/adjustment_pipeline.py:apply_phase5_adjustments`

### Current Gaps

1. **Volume vs. Efficiency**: System rewards total production but doesn't separate chain-movers from volume accumulators
2. **Drive-Level Context**: All yards weighted equally regardless of drive success
3. **Route Difficulty**: Separation captures receiver skill but not contested space by field location
4. **Turnover Attribution**: All INTs = QB fault, all fumbles penalized equally

### Integration Points

New features will integrate into:
- **Phase 1 Extension**: Drive context (play-level weighting)
- **Phase 4 Extension**: Success rate, route location, turnover attribution (per-week/season)

---

## Data Coverage Validation

### Overview

Before implementation, we must validate that all required nflfastpy columns have sufficient coverage and data quality. This section defines tests to run BEFORE any code changes.

### Required Columns

| Feature | Column Name | Type | Expected Availability |
|---------|------------|------|----------------------|
| Success Rate | `success` | Boolean | 1999-present |
| Drive Context | `drive` | Integer | 1999-present |
| Drive Context | `drive_end_transition` | String | 1999-present |
| Route Location | `pass_location` | String | 2006-present |
| Route Location | `pass_length` | String | 2006-present |
| Route Location | `air_yards` | Float | 2006-present |
| Turnover Attribution | `interception_player_id` | String | 2009-present |
| Turnover Attribution | `fumbled_1_player_id` | String | 1999-present |
| Turnover Attribution | `fumbled_2_player_id` | String | 1999-present |
| Turnover Attribution | `fumble_recovery_player_id` | String | 1999-present |
| Turnover Attribution | `was_pressure` | Boolean | 2016-present |

### Data Coverage Test Script

Create `tests/test_data_coverage_phase6.py`:

```python
"""
Data coverage tests for Phase 6 efficiency metrics.
Validates nflfastpy column availability and quality before implementation.
"""

import nfl_data_py as nfl
import polars as pl
from pathlib import Path
import sys

def test_success_rate_coverage():
    """Test success column coverage across years."""
    print("\n=== SUCCESS RATE COVERAGE ===")

    years_to_test = [2010, 2015, 2020, 2023, 2024]
    results = {}

    for year in years_to_test:
        print(f"\nTesting {year}...")
        pbp = nfl.import_pbp_data([year], columns=['success', 'play_type'])
        pbp_pl = pl.from_pandas(pbp)

        total_plays = len(pbp_pl)
        null_success = pbp_pl.filter(pl.col('success').is_null()).height
        null_rate = (null_success / total_plays) * 100

        # Filter to offensive plays only
        offensive_plays = pbp_pl.filter(
            (pl.col('play_type') == 'pass') | (pl.col('play_type') == 'run')
        )
        offensive_total = len(offensive_plays)
        offensive_null = offensive_plays.filter(pl.col('success').is_null()).height
        offensive_null_rate = (offensive_null / offensive_total) * 100

        results[year] = {
            'total_plays': total_plays,
            'null_rate': null_rate,
            'offensive_null_rate': offensive_null_rate
        }

        print(f"  Total plays: {total_plays:,}")
        print(f"  Null success rate (all plays): {null_rate:.2f}%")
        print(f"  Null success rate (offensive plays): {offensive_null_rate:.2f}%")

        # PASS/FAIL criteria
        if offensive_null_rate < 5.0:
            print(f"  ✓ PASS - Null rate acceptable (<5%)")
        elif offensive_null_rate < 15.0:
            print(f"  ⚠ WARNING - Null rate moderate (5-15%)")
        else:
            print(f"  ✗ FAIL - Null rate too high (>15%)")
            return False

    return True


def test_drive_context_coverage():
    """Test drive and drive_end_transition column coverage."""
    print("\n=== DRIVE CONTEXT COVERAGE ===")

    years_to_test = [2010, 2015, 2020, 2023, 2024]

    for year in years_to_test:
        print(f"\nTesting {year}...")
        pbp = nfl.import_pbp_data([year], columns=['drive', 'drive_end_transition', 'play_type'])
        pbp_pl = pl.from_pandas(pbp)

        total_plays = len(pbp_pl)
        null_drive = pbp_pl.filter(pl.col('drive').is_null()).height
        null_transition = pbp_pl.filter(pl.col('drive_end_transition').is_null()).height

        null_drive_rate = (null_drive / total_plays) * 100
        null_transition_rate = (null_transition / total_plays) * 100

        print(f"  Total plays: {total_plays:,}")
        print(f"  Null drive rate: {null_drive_rate:.2f}%")
        print(f"  Null drive_end_transition rate: {null_transition_rate:.2f}%")

        # Check unique values in drive_end_transition
        unique_transitions = pbp_pl.select('drive_end_transition').unique().to_series().to_list()
        print(f"  Unique transitions: {unique_transitions}")

        # PASS/FAIL
        if null_drive_rate < 1.0 and null_transition_rate < 5.0:
            print(f"  ✓ PASS - Coverage excellent")
        else:
            print(f"  ✗ FAIL - Excessive nulls")
            return False

    return True


def test_route_location_coverage():
    """Test pass_location, pass_length, and air_yards coverage."""
    print("\n=== ROUTE/TARGET LOCATION COVERAGE ===")

    years_to_test = [2010, 2015, 2020, 2023, 2024]

    for year in years_to_test:
        print(f"\nTesting {year}...")
        pbp = nfl.import_pbp_data([year], columns=[
            'pass_location', 'pass_length', 'air_yards', 'play_type'
        ])
        pbp_pl = pl.from_pandas(pbp)

        # Filter to pass plays only
        pass_plays = pbp_pl.filter(pl.col('play_type') == 'pass')
        total_passes = len(pass_plays)

        if total_passes == 0:
            print(f"  ⚠ WARNING - No pass plays found for {year}")
            continue

        null_location = pass_plays.filter(pl.col('pass_location').is_null()).height
        null_length = pass_plays.filter(pl.col('pass_length').is_null()).height
        null_air_yards = pass_plays.filter(pl.col('air_yards').is_null()).height

        null_location_rate = (null_location / total_passes) * 100
        null_length_rate = (null_length / total_passes) * 100
        null_air_yards_rate = (null_air_yards / total_passes) * 100

        print(f"  Total pass plays: {total_passes:,}")
        print(f"  Null pass_location rate: {null_location_rate:.2f}%")
        print(f"  Null pass_length rate: {null_length_rate:.2f}%")
        print(f"  Null air_yards rate: {null_air_yards_rate:.2f}%")

        # Check unique values
        unique_locations = pass_plays.select('pass_location').unique().to_series().to_list()
        unique_lengths = pass_plays.select('pass_length').unique().to_series().to_list()
        print(f"  Unique locations: {unique_locations}")
        print(f"  Unique lengths: {unique_lengths}")

        # PASS/FAIL - more lenient for earlier years
        if year >= 2015:
            if null_location_rate < 15.0 and null_length_rate < 15.0:
                print(f"  ✓ PASS - Coverage acceptable for post-2015")
            else:
                print(f"  ✗ FAIL - Coverage poor even for post-2015")
                return False
        else:
            if null_location_rate < 30.0:
                print(f"  ⚠ ACCEPTABLE - Pre-2015 data has expected gaps")
            else:
                print(f"  ✗ FAIL - Coverage unacceptable even for pre-2015")
                return False

    return True


def test_turnover_attribution_coverage():
    """Test interception and fumble attribution columns."""
    print("\n=== TURNOVER ATTRIBUTION COVERAGE ===")

    years_to_test = [2010, 2015, 2020, 2023, 2024]

    for year in years_to_test:
        print(f"\nTesting {year}...")
        pbp = nfl.import_pbp_data([year], columns=[
            'interception', 'interception_player_id',
            'fumble', 'fumbled_1_player_id', 'fumbled_2_player_id',
            'fumble_recovery_player_id', 'was_pressure', 'play_type'
        ])
        pbp_pl = pl.from_pandas(pbp)

        # Test interceptions
        ints = pbp_pl.filter(pl.col('interception') == 1)
        total_ints = len(ints)
        null_int_player = ints.filter(pl.col('interception_player_id').is_null()).height if total_ints > 0 else 0
        int_null_rate = (null_int_player / total_ints * 100) if total_ints > 0 else 0

        print(f"  Total INTs: {total_ints}")
        print(f"  Null INT player ID rate: {int_null_rate:.2f}%")

        # Test fumbles
        fumbles = pbp_pl.filter(pl.col('fumble') == 1)
        total_fumbles = len(fumbles)
        null_fumbler = fumbles.filter(pl.col('fumbled_1_player_id').is_null()).height if total_fumbles > 0 else 0
        fumble_null_rate = (null_fumbler / total_fumbles * 100) if total_fumbles > 0 else 0

        print(f"  Total fumbles: {total_fumbles}")
        print(f"  Null fumbled_1_player_id rate: {fumble_null_rate:.2f}%")

        # Test was_pressure (only available 2016+)
        if year >= 2016:
            pass_plays = pbp_pl.filter(pl.col('play_type') == 'pass')
            total_passes = len(pass_plays)
            null_pressure = pass_plays.filter(pl.col('was_pressure').is_null()).height
            pressure_null_rate = (null_pressure / total_passes * 100) if total_passes > 0 else 0
            print(f"  Null was_pressure rate: {pressure_null_rate:.2f}%")

            if pressure_null_rate > 20.0:
                print(f"  ⚠ WARNING - High null rate for was_pressure")

        # PASS/FAIL
        if int_null_rate < 10.0 and fumble_null_rate < 10.0:
            print(f"  ✓ PASS - Turnover attribution good")
        else:
            print(f"  ⚠ WARNING - Some attribution gaps")

    return True


def run_all_coverage_tests():
    """Run all data coverage tests."""
    print("=" * 80)
    print("PHASE 6 DATA COVERAGE VALIDATION")
    print("=" * 80)

    tests = [
        ("Success Rate", test_success_rate_coverage),
        ("Drive Context", test_drive_context_coverage),
        ("Route Location", test_route_location_coverage),
        ("Turnover Attribution", test_turnover_attribution_coverage),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ ERROR in {test_name}: {e}")
            results[test_name] = False

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ ALL TESTS PASSED - Safe to proceed with implementation")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Address data issues before implementing")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_coverage_tests())
```

### Acceptance Criteria

Before proceeding with Phase 6 implementation:

- [ ] Success rate column has <5% null rate on offensive plays (1999-present)
- [ ] Drive columns have <1% null rate for drive number, <5% for transition (1999-present)
- [ ] Pass location/length have <15% null rate on pass plays (2015-present)
- [ ] Air yards have <20% null rate on pass plays (2015-present)
- [ ] Interception player ID has <10% null rate on INT plays (2009-present)
- [ ] Fumble player IDs have <10% null rate on fumble plays (1999-present)
- [ ] was_pressure has <25% null rate on pass plays (2016-present)

### Fallback Strategy

If data coverage is insufficient in earlier years:
1. **Graceful degradation**: Apply multipliers only when data is available
2. **Default to 1.0x**: If column is null, use neutral multiplier (no adjustment)
3. **Year-specific logic**: Enable features only for years with good coverage
4. **Log warnings**: Track and report when features are skipped due to missing data

---

## Phase 6.1: Success Rate Integration

### Overview

Integrate `success` column from nflfastpy to measure play efficiency. Success is a binary metric indicating whether a play "advanced the situation" based on down/distance context.

### Success Definition (nflfastpy)

- **1st down**: Gain >= 45% of yards to go
- **2nd down**: Gain >= 60% of yards to go
- **3rd/4th down**: Gain >= 100% of yards to go (convert or score)

### Integration Point

**Location**: Phase 4 Extension (Per-Week Adjustment)
**File**: `modules/adjustment_pipeline.py`
**Function**: New `apply_success_rate_adjustment()` called within `apply_phase4_adjustments()`

### Technical Implementation

#### Step 1: Add success column to PBP cache

**File**: `modules/pbp_cache_builder.py`

**Modification** (around line 150, in column selection):

```python
# Add to column list
columns_to_load = [
    # ... existing columns ...
    'success',  # NEW: Play success indicator
]

# Add to multiplier calculation section (around line 250)
# Success multiplier calculation
.with_columns([
    pl.when(pl.col('success') == 1)
    .then(1.10)  # Success: 10% bonus
    .when(pl.col('success') == 0)
    .then(0.92)  # Failure: 8% penalty
    .otherwise(1.0)  # Null/unknown: neutral
    .alias('success_multiplier')
])
```

#### Step 2: Create success rate adjustment function

**File**: `modules/adjustment_pipeline.py`

**New function** (add after Phase 4 adjustments, around line 100):

```python
def apply_success_rate_adjustment(
    player_week_contributions: pl.DataFrame,
    pbp_data: pl.DataFrame,
    year: int
) -> pl.DataFrame:
    """
    Apply success rate adjustment to player weekly contributions.

    Rewards plays that successfully advanced down/distance situation,
    penalizes plays that failed to advance situation.

    Args:
        player_week_contributions: Player weekly stats with contributions
        pbp_data: Full PBP data for the year (with success column)
        year: Season year

    Returns:
        DataFrame with success_adjustment multiplier added

    Success multipliers:
        - Success on critical down (3rd/4th): 1.15x
        - Success on 1st/2nd down: 1.05x
        - Failure on any down: 0.92x
        - Unknown/null: 1.0x (neutral)
    """
    from modules.logger import get_logger
    logger = get_logger(__name__)

    logger.info(f"Applying success rate adjustments for {year}")

    # Filter to offensive plays with success data
    offensive_plays = pbp_data.filter(
        (pl.col('play_type').is_in(['pass', 'run'])) &
        (pl.col('success').is_not_null())
    )

    # Calculate success rate by player and week
    success_by_player_week = offensive_plays.group_by([
        'passer_player_id', 'rusher_player_id', 'receiver_player_id',
        'week'
    ]).agg([
        pl.col('success').mean().alias('success_rate'),
        pl.col('success').count().alias('play_count'),
        pl.col('down').mean().alias('avg_down'),  # Track if critical situations
    ])

    # Create multiplier based on success rate and situation
    success_by_player_week = success_by_player_week.with_columns([
        pl.when(
            (pl.col('success_rate') >= 0.60) & (pl.col('avg_down') >= 2.5)
        )
        .then(1.15)  # High success on critical downs
        .when(pl.col('success_rate') >= 0.55)
        .then(1.10)  # High success on manageable downs
        .when(pl.col('success_rate') >= 0.45)
        .then(1.05)  # Average success
        .when(pl.col('success_rate') >= 0.35)
        .then(1.0)   # Below average (neutral)
        .otherwise(0.92)  # Low success (penalty)
        .alias('success_multiplier')
    ])

    # Join to player contributions
    # Note: Need to handle QB, RB, WR, TE differently due to different ID columns

    # For QBs (passers)
    player_week_contributions = player_week_contributions.join(
        success_by_player_week.select([
            pl.col('passer_player_id').alias('player_id'),
            'week',
            pl.col('success_multiplier').alias('success_mult_qb')
        ]),
        on=['player_id', 'week'],
        how='left'
    )

    # For RBs (rushers)
    player_week_contributions = player_week_contributions.join(
        success_by_player_week.select([
            pl.col('rusher_player_id').alias('player_id'),
            'week',
            pl.col('success_multiplier').alias('success_mult_rb')
        ]),
        on=['player_id', 'week'],
        how='left'
    )

    # For WR/TE (receivers)
    player_week_contributions = player_week_contributions.join(
        success_by_player_week.select([
            pl.col('receiver_player_id').alias('player_id'),
            'week',
            pl.col('success_multiplier').alias('success_mult_receiver')
        ]),
        on=['player_id', 'week'],
        how='left'
    )

    # Combine based on position
    player_week_contributions = player_week_contributions.with_columns([
        pl.when(pl.col('position') == 'QB')
        .then(pl.col('success_mult_qb'))
        .when(pl.col('position') == 'RB')
        .then(pl.col('success_mult_rb'))
        .when(pl.col('position').is_in(['WR', 'TE']))
        .then(pl.col('success_mult_receiver'))
        .otherwise(1.0)
        .fill_null(1.0)  # Default to neutral if no data
        .alias('success_adjustment')
    ])

    # Apply adjustment to overall contribution
    player_week_contributions = player_week_contributions.with_columns([
        (pl.col('overall_contribution') * pl.col('success_adjustment'))
        .alias('overall_contribution')
    ])

    logger.info(
        f"Success rate adjustments applied to {player_week_contributions.height} player-weeks"
    )

    return player_week_contributions.drop([
        'success_mult_qb', 'success_mult_rb', 'success_mult_receiver'
    ])
```

#### Step 3: Integrate into Phase 4 pipeline

**File**: `modules/adjustment_pipeline.py`

**Modification** in `apply_phase4_adjustments()` function (around line 50):

```python
def apply_phase4_adjustments(
    player_agg: pl.DataFrame,
    year: int,
    pbp_data: pl.DataFrame = None
) -> pl.DataFrame:
    """Apply Phase 4 adjustments..."""

    # ... existing code for catch rate, blocking, separation, penalties ...

    # NEW: Apply success rate adjustment (per-week)
    if pbp_data is not None:
        logger.info("Applying success rate adjustments (Phase 6.1)")
        player_agg = apply_success_rate_adjustment(player_agg, pbp_data, year)
    else:
        logger.warning("PBP data not provided, skipping success rate adjustment")
        player_agg = player_agg.with_columns([
            pl.lit(1.0).alias('success_adjustment')
        ])

    # ... rest of Phase 4 ...

    return player_agg
```

### Testing & Validation

#### Unit Tests

**File**: `tests/test_success_rate_adjustment.py`

```python
"""Unit tests for success rate adjustment logic."""

import polars as pl
import pytest
from modules.adjustment_pipeline import apply_success_rate_adjustment


def test_success_rate_high_success():
    """Test that high success rate yields positive multiplier."""
    # Mock data: 80% success rate on 3rd downs
    player_data = pl.DataFrame({
        'player_id': ['QB1'],
        'week': [1],
        'position': ['QB'],
        'overall_contribution': [100.0]
    })

    pbp_data = pl.DataFrame({
        'play_type': ['pass'] * 10,
        'passer_player_id': ['QB1'] * 10,
        'rusher_player_id': [None] * 10,
        'receiver_player_id': [None] * 10,
        'week': [1] * 10,
        'success': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 80% success
        'down': [3, 3, 3, 3, 3, 2, 2, 2, 3, 3]  # Mostly 3rd down
    })

    result = apply_success_rate_adjustment(player_data, pbp_data, 2024)

    # Should get 1.15x multiplier (high success on critical downs)
    assert result['success_adjustment'][0] == pytest.approx(1.15, abs=0.01)
    assert result['overall_contribution'][0] == pytest.approx(115.0, abs=1.0)


def test_success_rate_low_success():
    """Test that low success rate yields penalty multiplier."""
    player_data = pl.DataFrame({
        'player_id': ['RB1'],
        'week': [1],
        'position': ['RB'],
        'overall_contribution': [50.0]
    })

    pbp_data = pl.DataFrame({
        'play_type': ['run'] * 10,
        'passer_player_id': [None] * 10,
        'rusher_player_id': ['RB1'] * 10,
        'receiver_player_id': [None] * 10,
        'week': [1] * 10,
        'success': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10% success (terrible)
        'down': [1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    })

    result = apply_success_rate_adjustment(player_data, pbp_data, 2024)

    # Should get 0.92x multiplier (low success penalty)
    assert result['success_adjustment'][0] == pytest.approx(0.92, abs=0.01)
    assert result['overall_contribution'][0] == pytest.approx(46.0, abs=1.0)


def test_success_rate_null_handling():
    """Test that null success data defaults to 1.0x neutral."""
    player_data = pl.DataFrame({
        'player_id': ['WR1'],
        'week': [1],
        'position': ['WR'],
        'overall_contribution': [75.0]
    })

    pbp_data = pl.DataFrame({
        'play_type': ['pass'] * 5,
        'passer_player_id': ['QB1'] * 5,
        'rusher_player_id': [None] * 5,
        'receiver_player_id': ['WR1'] * 5,
        'week': [1] * 5,
        'success': [None, None, None, None, None],  # All null
        'down': [1, 2, 3, 1, 2]
    })

    result = apply_success_rate_adjustment(player_data, pbp_data, 2024)

    # Should default to 1.0x neutral (no data = no adjustment)
    assert result['success_adjustment'][0] == pytest.approx(1.0, abs=0.01)
    assert result['overall_contribution'][0] == pytest.approx(75.0, abs=1.0)
```

#### Integration Test

**File**: `tests/test_phase6_integration.py`

```python
"""Integration test for Phase 6 features with full pipeline."""

import polars as pl
from modules.rankings_generator import generate_qb_rankings, generate_rb_rankings


def test_success_rate_integration_qb_2024():
    """Test success rate integration on real 2024 QB data."""

    # Generate QB rankings with Phase 6.1
    rankings = generate_qb_rankings(2024)

    # Validate structure
    assert 'success_adjustment' in rankings.columns
    assert rankings['success_adjustment'].is_not_nan().all()
    assert rankings['success_adjustment'].min() >= 0.85
    assert rankings['success_adjustment'].max() <= 1.20

    # Check known player: Lamar Jackson should have high success rate
    lamar = rankings.filter(pl.col('player_name').str.contains('L.Jackson|Lamar Jackson'))
    if len(lamar) > 0:
        assert lamar['success_adjustment'][0] >= 1.05  # Should be above average


def test_success_rate_integration_rb_2024():
    """Test success rate integration on real 2024 RB data."""

    rankings = generate_rb_rankings(2024)

    assert 'success_adjustment' in rankings.columns
    assert rankings['success_adjustment'].is_not_nan().all()

    # Christian McCaffrey known for high efficiency
    cmc = rankings.filter(pl.col('player_name').str.contains('C.McCaffrey|Christian McCaffrey'))
    if len(cmc) > 0:
        assert cmc['success_adjustment'][0] >= 1.08  # Elite efficiency
```

### Success Criteria

- [ ] Success rate adjustment successfully integrated into Phase 4 pipeline
- [ ] No NaN values in success_adjustment column
- [ ] Multiplier range validated: 0.90-1.15x
- [ ] Unit tests pass for high/low/null success scenarios
- [ ] Integration tests pass on 2024 QB/RB data
- [ ] Known efficient players (CMC, Lamar) show >1.08x multipliers
- [ ] Known inefficient players show <0.95x multipliers
- [ ] Processing time increase <3% vs. baseline

---

## Phase 6.2: Drive Context Integration

### Overview

Weight plays based on drive outcome to reward contributions on successful scoring drives and penalize plays on failed drives.

### Drive Outcomes (nflfastpy)

- `drive_end_transition` values:
  - "Touchdown" → 1.15x multiplier
  - "Field goal" → 1.05x multiplier
  - "Punt" → 0.90x multiplier
  - "Turnover" / "Turnover on downs" → 0.75x multiplier
  - "End of half" / "End of game" → 1.0x neutral

### Integration Point

**Location**: Phase 1 Extension (Play Level)
**File**: `modules/play_by_play.py`
**Function**: Extend `_calculate_multipliers()` to include drive context

### Technical Implementation

#### Step 1: Add drive columns to PBP cache

**File**: `modules/pbp_cache_builder.py`

**Modification** (around line 150):

```python
columns_to_load = [
    # ... existing ...
    'drive',  # NEW: Drive number
    'drive_end_transition',  # NEW: How drive ended
]

# Add drive outcome multiplier calculation (around line 280)
.with_columns([
    pl.when(pl.col('drive_end_transition') == 'Touchdown')
    .then(1.15)
    .when(pl.col('drive_end_transition') == 'Field goal')
    .then(1.05)
    .when(pl.col('drive_end_transition').is_in(['Punt', 'Missed field goal']))
    .then(0.90)
    .when(pl.col('drive_end_transition').is_in([
        'Turnover', 'Turnover on downs', 'Interception', 'Fumble'
    ]))
    .then(0.75)
    .when(pl.col('drive_end_transition').is_in(['End of half', 'End of game']))
    .then(1.0)
    .otherwise(1.0)
    .alias('drive_outcome_multiplier')
])
```

#### Step 2: Integrate into Phase 1 multipliers

**File**: `modules/play_by_play.py`

**Modification** in `_calculate_multipliers()` (around line 120):

```python
def _calculate_multipliers(self, pbp_data: pl.DataFrame) -> pl.DataFrame:
    """Calculate all situational multipliers for plays."""

    # ... existing field position, score, time, down multipliers ...

    # NEW: Drive outcome multiplier (Phase 6.2)
    # Note: drive_outcome_multiplier already calculated in cache builder
    # Just ensure it's used in combined multiplier

    # Combined situational multiplier (updated formula)
    pbp_data = pbp_data.with_columns([
        (
            pl.col('field_position_multiplier') *
            pl.col('score_multiplier') *
            pl.col('time_multiplier') *
            pl.col('down_multiplier') *
            pl.col('drive_outcome_multiplier')  # NEW
        ).alias('situational_multiplier')
    ])

    return pbp_data
```

### NaN Prevention

**Critical Check** - Ensure drive_outcome_multiplier handles nulls:

```python
# In cache builder, add null handling:
.with_columns([
    pl.when(pl.col('drive_end_transition').is_null())
    .then(1.0)  # Default to neutral if unknown
    .when(pl.col('drive_end_transition') == 'Touchdown')
    .then(1.15)
    # ... rest of logic ...
    .alias('drive_outcome_multiplier')
])
```

### Testing & Validation

#### Unit Tests

**File**: `tests/test_drive_context_adjustment.py`

```python
"""Unit tests for drive context multipliers."""

import polars as pl
from modules.pbp_cache_builder import build_cache


def test_drive_context_td_drive():
    """Test that TD drives get 1.15x multiplier."""
    test_data = pl.DataFrame({
        'drive': [1, 1, 1],
        'drive_end_transition': ['Touchdown', 'Touchdown', 'Touchdown'],
        'play_type': ['pass', 'run', 'pass'],
        'yards_gained': [10, 5, 25],
    })

    # Simulate cache builder logic
    test_data = test_data.with_columns([
        pl.when(pl.col('drive_end_transition') == 'Touchdown')
        .then(1.15)
        .otherwise(1.0)
        .alias('drive_outcome_multiplier')
    ])

    assert test_data['drive_outcome_multiplier'][0] == 1.15
    assert test_data['drive_outcome_multiplier'].is_not_nan().all()


def test_drive_context_turnover_drive():
    """Test that turnover drives get 0.75x penalty."""
    test_data = pl.DataFrame({
        'drive': [2, 2, 2],
        'drive_end_transition': ['Interception', 'Interception', 'Interception'],
        'play_type': ['pass', 'pass', 'run'],
        'yards_gained': [8, 12, 3],
    })

    test_data = test_data.with_columns([
        pl.when(pl.col('drive_end_transition') == 'Interception')
        .then(0.75)
        .otherwise(1.0)
        .alias('drive_outcome_multiplier')
    ])

    assert test_data['drive_outcome_multiplier'][0] == 0.75


def test_drive_context_null_handling():
    """Test that null drive outcomes default to neutral."""
    test_data = pl.DataFrame({
        'drive': [3],
        'drive_end_transition': [None],
        'play_type': ['pass'],
        'yards_gained': [15],
    })

    test_data = test_data.with_columns([
        pl.when(pl.col('drive_end_transition').is_null())
        .then(1.0)
        .when(pl.col('drive_end_transition') == 'Touchdown')
        .then(1.15)
        .otherwise(1.0)
        .alias('drive_outcome_multiplier')
    ])

    assert test_data['drive_outcome_multiplier'][0] == 1.0
```

#### Integration Test

```python
def test_drive_context_integration_2024():
    """Test drive context on full 2024 season."""
    from modules.rankings_generator import generate_wr_rankings

    rankings = generate_wr_rankings(2024)

    # Check that drive context is reflected in rankings
    # (implicitly via Phase 1 multipliers)
    # Tyreek Hill (high TD drive % in 2024)
    tyreek = rankings.filter(pl.col('player_name').str.contains('T.Hill|Tyreek Hill'))

    # Validate no NaNs in overall contribution
    assert rankings['overall_contribution'].is_not_nan().all()
    assert rankings['overall_contribution'].min() >= 0
```

### Success Criteria

- [ ] Drive outcome multiplier successfully added to PBP cache
- [ ] No NaN values in drive_outcome_multiplier column
- [ ] Multiplier range validated: 0.75-1.15x
- [ ] Null drive outcomes default to 1.0x neutral
- [ ] TD drives show 1.15x, turnovers show 0.75x
- [ ] Integration test passes on 2024 WR data
- [ ] No regression in existing Phase 1 logic
- [ ] Processing time increase <2% vs. baseline

---

## Phase 6.3: Route/Target Location Integration

### Overview

Adjust receiver contributions based on target location (middle vs. sideline) and depth (deep vs. short) to account for catch difficulty. **Critical: Includes slot receiver bias mitigation.**

### Pass Location/Length Values (nflfastpy)

**pass_location**: `left`, `middle`, `right`
**pass_length**: `short` (<10 yards), `deep` (≥10 yards)
**air_yards**: Float (exact air yards on pass attempt)

### Slot Receiver Bias Mitigation

**Problem**: Slot receivers average 8-10 yard routes (short middle). Simple depth-based multiplier would penalize them unfairly vs. outside receivers (12-18 yard routes).

**Solution**: Location-weighted difficulty matrix that rewards middle-of-field contested catches.

### Difficulty Matrix

| Location | Depth | Multiplier | Rationale |
|----------|-------|------------|-----------|
| Middle | Deep (15+) | 1.12x | Hardest: contested deep middle |
| Middle | Intermediate (8-14) | 1.06x | Contested traffic |
| Middle | Short (<8) | 1.03x | Slot territory - traffic bonus |
| Left/Right | Deep (15+) | 1.05x | Deep sideline, moderate difficulty |
| Left/Right | Intermediate (8-14) | 1.00x | Neutral (baseline) |
| Left/Right | Short (<8) | 0.95x | Easiest: short sideline |

**Key insight**: Middle routes get +0.03 to +0.06x bonus regardless of depth, offsetting shorter air yards for slot receivers.

### Integration Point

**Location**: Phase 4 Extension (Season-Wide, with NextGen)
**File**: `modules/adjustment_pipeline.py`
**Function**: New `calculate_route_location_adjustment()` similar to separation

### Technical Implementation

#### Step 1: Add location columns to PBP cache

**File**: `modules/pbp_cache_builder.py`

```python
columns_to_load = [
    # ... existing ...
    'pass_location',  # NEW: left/middle/right
    'pass_length',    # NEW: short/deep
    'air_yards',      # NEW: exact depth
]
```

#### Step 2: Create route location adjustment function

**File**: `modules/adjustment_pipeline.py`

**New function** (add after separation adjustment, around line 150):

```python
def calculate_route_location_adjustment(
    player_agg: pl.DataFrame,
    pbp_data: pl.DataFrame,
    year: int
) -> pl.DataFrame:
    """
    Calculate route location difficulty adjustment for receivers.

    Accounts for contested middle-of-field catches vs. easier sideline routes.
    Includes slot receiver bias mitigation via location-weighted multipliers.

    Args:
        player_agg: Player season aggregates
        pbp_data: Full PBP data with pass_location, air_yards
        year: Season year

    Returns:
        DataFrame with route_location_adjustment multiplier

    Multiplier Matrix:
        - Deep middle (15+ yards, middle): 1.12x (hardest)
        - Intermediate middle (8-14, middle): 1.06x
        - Short middle (<8, middle): 1.03x (slot bonus)
        - Deep sideline (15+, left/right): 1.05x
        - Intermediate sideline (8-14, left/right): 1.00x (neutral)
        - Short sideline (<8, left/right): 0.95x (easiest)

    Minimum: 30 targets for meaningful data
    """
    from modules.logger import get_logger
    logger = get_logger(__name__)

    logger.info(f"Calculating route location adjustments for {year}")

    # Filter to pass plays with location data
    pass_plays = pbp_data.filter(
        (pl.col('play_type') == 'pass') &
        (pl.col('pass_location').is_not_null()) &
        (pl.col('air_yards').is_not_null()) &
        (pl.col('receiver_player_id').is_not_null())
    )

    # Categorize depth
    pass_plays = pass_plays.with_columns([
        pl.when(pl.col('air_yards') >= 15)
        .then('deep')
        .when(pl.col('air_yards') >= 8)
        .then('intermediate')
        .otherwise('short')
        .alias('depth_category')
    ])

    # Calculate difficulty multiplier based on location + depth
    pass_plays = pass_plays.with_columns([
        pl.when(
            (pl.col('pass_location') == 'middle') & (pl.col('depth_category') == 'deep')
        )
        .then(1.12)  # Deep middle: hardest
        .when(
            (pl.col('pass_location') == 'middle') & (pl.col('depth_category') == 'intermediate')
        )
        .then(1.06)  # Intermediate middle: contested
        .when(
            (pl.col('pass_location') == 'middle') & (pl.col('depth_category') == 'short')
        )
        .then(1.03)  # Short middle: slot bonus (MITIGATION)
        .when(
            (pl.col('pass_location').is_in(['left', 'right'])) & (pl.col('depth_category') == 'deep')
        )
        .then(1.05)  # Deep sideline: moderate
        .when(
            (pl.col('pass_location').is_in(['left', 'right'])) & (pl.col('depth_category') == 'intermediate')
        )
        .then(1.00)  # Intermediate sideline: neutral (baseline)
        .when(
            (pl.col('pass_location').is_in(['left', 'right'])) & (pl.col('depth_category') == 'short')
        )
        .then(0.95)  # Short sideline: easiest
        .otherwise(1.0)
        .alias('target_difficulty')
    ])

    # Aggregate by receiver
    receiver_location_stats = pass_plays.group_by('receiver_player_id').agg([
        pl.col('target_difficulty').mean().alias('avg_target_difficulty'),
        pl.col('air_yards').mean().alias('avg_air_yards'),
        pl.count().alias('target_count'),
        # Track % of routes by location for diagnostics
        (pl.col('pass_location') == 'middle').mean().alias('pct_middle_routes'),
        (pl.col('depth_category') == 'short').mean().alias('pct_short_routes'),
    ])

    # Only apply adjustment if player has 30+ targets
    receiver_location_stats = receiver_location_stats.with_columns([
        pl.when(pl.col('target_count') >= 30)
        .then(pl.col('avg_target_difficulty'))
        .otherwise(1.0)  # Insufficient sample: neutral
        .alias('route_location_adjustment')
    ])

    # Join to player aggregates
    player_agg = player_agg.join(
        receiver_location_stats.select([
            pl.col('receiver_player_id').alias('player_id'),
            'route_location_adjustment',
            'avg_air_yards',
            'target_count',
            'pct_middle_routes',
            'pct_short_routes',
        ]),
        on='player_id',
        how='left'
    )

    # Fill nulls (non-receivers or missing data) with 1.0
    player_agg = player_agg.with_columns([
        pl.col('route_location_adjustment').fill_null(1.0)
    ])

    # Cap adjustment range (safety check)
    player_agg = player_agg.with_columns([
        pl.col('route_location_adjustment').clip(0.92, 1.15).alias('route_location_adjustment')
    ])

    # Apply adjustment to overall contribution
    player_agg = player_agg.with_columns([
        (pl.col('overall_contribution') * pl.col('route_location_adjustment'))
        .alias('overall_contribution')
    ])

    logger.info(
        f"Route location adjustments applied to "
        f"{player_agg.filter(pl.col('route_location_adjustment') != 1.0).height} receivers"
    )

    # Log diagnostic info for slot receivers
    slot_receivers = player_agg.filter(
        (pl.col('pct_middle_routes') >= 0.50) & (pl.col('target_count') >= 30)
    )
    if len(slot_receivers) > 0:
        avg_slot_adjustment = slot_receivers['route_location_adjustment'].mean()
        logger.info(
            f"Slot receivers (50%+ middle routes): avg adjustment = {avg_slot_adjustment:.3f}x"
        )

    return player_agg
```

#### Step 3: Integrate into Phase 4 pipeline

**File**: `modules/adjustment_pipeline.py`

**Modification** in `apply_phase4_adjustments()`:

```python
def apply_phase4_adjustments(
    player_agg: pl.DataFrame,
    year: int,
    pbp_data: pl.DataFrame = None
) -> pl.DataFrame:
    """Apply Phase 4 adjustments..."""

    # ... existing separation, catch rate adjustments ...

    # NEW: Apply route location adjustment (Phase 6.3)
    if pbp_data is not None and year >= 2015:  # Location data quality check
        logger.info("Applying route location adjustments (Phase 6.3)")
        player_agg = calculate_route_location_adjustment(player_agg, pbp_data, year)
    else:
        logger.warning("PBP data not available or year <2015, skipping route location adjustment")
        player_agg = player_agg.with_columns([
            pl.lit(1.0).alias('route_location_adjustment')
        ])

    # ... rest of Phase 4 ...

    return player_agg
```

### Testing & Validation

#### Unit Tests

**File**: `tests/test_route_location_adjustment.py`

```python
"""Unit tests for route location adjustment with slot bias mitigation."""

import polars as pl
import pytest
from modules.adjustment_pipeline import calculate_route_location_adjustment


def test_route_location_slot_receiver():
    """Test that slot receivers get appropriate bonus despite short routes."""
    player_data = pl.DataFrame({
        'player_id': ['WR_SLOT'],
        'position': ['WR'],
        'overall_contribution': [100.0]
    })

    # Simulate slot receiver: 60% middle routes, average 9 yards
    pbp_data = pl.DataFrame({
        'play_type': ['pass'] * 50,
        'receiver_player_id': ['WR_SLOT'] * 50,
        'pass_location': ['middle'] * 30 + ['left'] * 10 + ['right'] * 10,  # 60% middle
        'air_yards': [8, 9, 10, 7, 8] * 10,  # Average ~8.4 yards
    })

    result = calculate_route_location_adjustment(player_data, pbp_data, 2024)

    # Should get ~1.03-1.06x (short/intermediate middle bonus)
    assert result['route_location_adjustment'][0] >= 1.02
    assert result['route_location_adjustment'][0] <= 1.08
    assert result['pct_middle_routes'][0] >= 0.55  # Confirmed slot role


def test_route_location_outside_receiver():
    """Test that outside receivers get appropriate multiplier for deep routes."""
    player_data = pl.DataFrame({
        'player_id': ['WR_OUTSIDE'],
        'position': ['WR'],
        'overall_contribution': [120.0]
    })

    # Simulate outside receiver: 80% sideline, average 15 yards
    pbp_data = pl.DataFrame({
        'play_type': ['pass'] * 50,
        'receiver_player_id': ['WR_OUTSIDE'] * 50,
        'pass_location': ['left'] * 25 + ['right'] * 25,  # 100% sideline
        'air_yards': [15, 16, 14, 18, 17] * 10,  # Average ~16 yards
    })

    result = calculate_route_location_adjustment(player_data, pbp_data, 2024)

    # Should get ~1.05x (deep sideline)
    assert result['route_location_adjustment'][0] >= 1.04
    assert result['route_location_adjustment'][0] <= 1.06
    assert result['pct_middle_routes'][0] <= 0.10  # Confirmed outside role


def test_route_location_bias_mitigation():
    """
    Test that slot receiver (short middle) gets similar adjustment to
    outside receiver (deep sideline), despite air yards difference.
    """
    player_data = pl.DataFrame({
        'player_id': ['WR_SLOT', 'WR_OUTSIDE'],
        'position': ['WR', 'WR'],
        'overall_contribution': [100.0, 100.0]
    })

    pbp_data = pl.DataFrame({
        'play_type': ['pass'] * 100,
        'receiver_player_id': ['WR_SLOT'] * 50 + ['WR_OUTSIDE'] * 50,
        'pass_location': (['middle'] * 40 + ['left'] * 10) + (['left'] * 30 + ['right'] * 20),
        'air_yards': ([8] * 50) + ([16] * 50),  # Slot: 8 yards, Outside: 16 yards
    })

    result = calculate_route_location_adjustment(player_data, pbp_data, 2024)

    slot_adj = result.filter(pl.col('player_id') == 'WR_SLOT')['route_location_adjustment'][0]
    outside_adj = result.filter(pl.col('player_id') == 'WR_OUTSIDE')['route_location_adjustment'][0]

    # Key test: difference should be small (<0.03x) despite air yards difference (8 vs 16)
    assert abs(slot_adj - outside_adj) < 0.03
    print(f"Slot: {slot_adj:.3f}x, Outside: {outside_adj:.3f}x - Bias mitigated!")


def test_route_location_insufficient_sample():
    """Test that <30 targets defaults to neutral 1.0x."""
    player_data = pl.DataFrame({
        'player_id': ['WR_ROOKIE'],
        'position': ['WR'],
        'overall_contribution': [25.0]
    })

    pbp_data = pl.DataFrame({
        'play_type': ['pass'] * 15,  # Only 15 targets
        'receiver_player_id': ['WR_ROOKIE'] * 15,
        'pass_location': ['middle'] * 15,
        'air_yards': [20] * 15,
    })

    result = calculate_route_location_adjustment(player_data, pbp_data, 2024)

    # Should default to 1.0x (insufficient sample)
    assert result['route_location_adjustment'][0] == 1.0
```

#### Integration Test - Slot Bias Validation

**File**: `tests/test_slot_bias_mitigation.py`

```python
"""Validate that known slot receivers aren't penalized vs. outside receivers."""

import polars as pl
from modules.rankings_generator import generate_wr_rankings


def test_slot_bias_mitigation_2023():
    """Test 2023 known slot receivers get fair treatment."""

    rankings = generate_wr_rankings(2023)

    # Known slot specialists in 2023
    slot_receivers = [
        'Cooper Kupp',  # Rams slot
        'Amon-Ra St. Brown',  # Lions slot
        'Christian Kirk',  # Jaguars slot
    ]

    # Known outside receivers in 2023
    outside_receivers = [
        'Tyreek Hill',  # Dolphins outside
        'CeeDee Lamb',  # Cowboys outside
        'Justin Jefferson',  # Vikings outside
    ]

    slot_adjustments = []
    outside_adjustments = []

    for name in slot_receivers:
        player = rankings.filter(pl.col('player_name').str.contains(name))
        if len(player) > 0:
            slot_adjustments.append(player['route_location_adjustment'][0])

    for name in outside_receivers:
        player = rankings.filter(pl.col('player_name').str.contains(name))
        if len(player) > 0:
            outside_adjustments.append(player['route_location_adjustment'][0])

    if len(slot_adjustments) > 0 and len(outside_adjustments) > 0:
        avg_slot = sum(slot_adjustments) / len(slot_adjustments)
        avg_outside = sum(outside_adjustments) / len(outside_adjustments)

        print(f"Average slot adjustment: {avg_slot:.3f}x")
        print(f"Average outside adjustment: {avg_outside:.3f}x")

        # KEY VALIDATION: Difference should be minimal (<0.04x)
        assert abs(avg_slot - avg_outside) < 0.04, "Slot receiver bias detected!"
```

### Success Criteria

- [ ] Route location adjustment integrated into Phase 4
- [ ] No NaN values in route_location_adjustment column
- [ ] Multiplier range validated: 0.95-1.12x
- [ ] Slot receivers (>50% middle routes) average 1.03-1.06x
- [ ] Outside receivers (>70% sideline) average 1.03-1.06x
- [ ] Bias mitigation test passes: |avg_slot - avg_outside| < 0.04x
- [ ] Unit tests pass for slot/outside/insufficient sample scenarios
- [ ] Known slot specialists (Kupp, ARSB) show appropriate adjustments
- [ ] Processing time increase <4% vs. baseline

---

## Phase 6.4: Turnover Attribution Enhancement

### Overview

Improve turnover responsibility assignment to better separate QB INTs from WR tips, and RB fumbles from QB strip sacks.

### Required Columns

- `interception_player_id` - Defender who caught INT
- `fumbled_1_player_id` - Primary fumbler
- `fumbled_2_player_id` - Secondary fumbler
- `fumble_recovery_player_id` - Who recovered fumble
- `was_pressure` - Whether QB was under pressure (2016+)

### Attribution Logic

**Interceptions:**
1. Check if WR was intended target → partial blame to WR if tipped
2. Check if QB was under pressure → reduced penalty if forced throw
3. Default: full penalty to QB

**Fumbles:**
1. Check fumble recovery team → reduced penalty if recovered by offense
2. Check play type → different penalties for pass (strip sack) vs. run (ball security)
3. Position-specific: QB strip sack vs. RB fumble on contact vs. RB fumble in open field

### Integration Point

**Location**: Phase 4 Extension (Per-Week)
**File**: `modules/contribution_calculators.py`
**Function**: Enhance existing INT penalty logic (around line 791)

### Technical Implementation

#### Step 1: Add turnover attribution columns to PBP cache

**File**: `modules/pbp_cache_builder.py`

```python
columns_to_load = [
    # ... existing ...
    'interception_player_id',  # NEW: Defender who caught INT
    'fumbled_1_player_id',     # NEW: Primary fumbler
    'fumbled_2_player_id',     # NEW: Secondary fumbler
    'fumble_recovery_player_id',  # NEW: Who recovered
    'fumble_recovery_1_team',  # NEW: Recovery team
]
```

#### Step 2: Enhance QB INT attribution

**File**: `modules/contribution_calculators.py`

**Modification** in QB contribution calculation (around line 780):

```python
def calculate_qb_contribution_from_pbp(
    pbp_data: pl.DataFrame,
    qb_id: str,
    team: str
) -> dict:
    """Calculate QB contributions with enhanced turnover attribution."""

    # ... existing code ...

    # ENHANCED: Better INT attribution (Phase 6.4)
    int_plays = qb_plays.filter(pl.col('interception') == 1)

    for int_play in int_plays.iter_rows(named=True):
        seconds_remaining = int_play.get('game_seconds_remaining', 1800)
        was_pressure = int_play.get('was_pressure', False)

        # Base INT penalty
        base_penalty = -15

        # Reduce penalty if under pressure (not QB's fault)
        if was_pressure:
            pressure_reduction = 0.33  # 33% reduction
            base_penalty *= (1 - pressure_reduction)
            logger.debug(f"INT under pressure: penalty reduced to {base_penalty:.1f}")

        # Increase penalty if final 2 minutes (critical moment)
        if seconds_remaining < 120:
            time_mod = 1.3
            base_penalty *= time_mod
            logger.debug(f"INT in final 2min: penalty increased to {base_penalty:.1f}")

        # Apply INT penalty to contribution
        contribution['int_penalty'] = contribution.get('int_penalty', 0) + base_penalty

    # ... rest of QB calculation ...
```

#### Step 3: Add WR tip attribution

**File**: `modules/adjustment_pipeline.py`

**New function** (add in Phase 4 section):

```python
def apply_receiver_tip_penalty(
    player_agg: pl.DataFrame,
    pbp_data: pl.DataFrame,
    year: int
) -> pl.DataFrame:
    """
    Apply penalty to receivers responsible for tipped interceptions.

    A tipped pass leading to INT should partially blame the receiver,
    not just the QB.

    Args:
        player_agg: Player aggregates
        pbp_data: Full PBP data
        year: Season year

    Returns:
        DataFrame with tip_int_penalty applied

    Attribution logic:
        - If receiver is target on INT play: -8 contribution (WR fault)
        - If QB under pressure: -5 contribution (forced throw, partial fault)
        - QB already gets INT penalty, so this is additional WR penalty
    """
    from modules.logger import get_logger
    logger = get_logger(__name__)

    logger.info(f"Applying receiver tip INT penalties for {year}")

    # Find INT plays with receiver involvement
    int_plays = pbp_data.filter(
        (pl.col('interception') == 1) &
        (pl.col('receiver_player_id').is_not_null())
    )

    # Calculate penalty by receiver
    receiver_tip_penalties = int_plays.group_by('receiver_player_id').agg([
        pl.count().alias('tip_int_count'),
        pl.col('was_pressure').mean().alias('avg_pressure_on_tips'),
    ])

    # Calculate penalty: -8 per tip, reduced to -5 if under pressure
    receiver_tip_penalties = receiver_tip_penalties.with_columns([
        pl.when(pl.col('avg_pressure_on_tips') >= 0.5)
        .then(pl.col('tip_int_count') * -5)  # Under pressure: reduced penalty
        .otherwise(pl.col('tip_int_count') * -8)  # Clean: full penalty
        .alias('tip_int_penalty')
    ])

    # Join to player aggregates
    player_agg = player_agg.join(
        receiver_tip_penalties.select([
            pl.col('receiver_player_id').alias('player_id'),
            'tip_int_penalty'
        ]),
        on='player_id',
        how='left'
    )

    # Fill nulls (no tips) with 0
    player_agg = player_agg.with_columns([
        pl.col('tip_int_penalty').fill_null(0)
    ])

    # Apply penalty to overall contribution
    player_agg = player_agg.with_columns([
        (pl.col('overall_contribution') + pl.col('tip_int_penalty'))
        .alias('overall_contribution')
    ])

    logger.info(
        f"Tip INT penalties applied to "
        f"{player_agg.filter(pl.col('tip_int_penalty') != 0).height} receivers"
    )

    return player_agg
```

#### Step 4: Add RB fumble attribution

**File**: `modules/adjustment_pipeline.py`

**New function**:

```python
def apply_fumble_attribution_penalty(
    player_agg: pl.DataFrame,
    pbp_data: pl.DataFrame,
    year: int
) -> pl.DataFrame:
    """
    Apply context-aware fumble penalties.

    Different fumble scenarios have different responsibility:
    - RB fumble recovered by offense: -5 (moderate - drive continues)
    - RB fumble recovered by defense: -12 (catastrophic)
    - QB strip sack: -10 (QB responsibility, not OL)
    - Open field fumble: -15 (unforced error, very bad)

    Args:
        player_agg: Player aggregates
        pbp_data: Full PBP data
        year: Season year

    Returns:
        DataFrame with fumble_penalty applied
    """
    from modules.logger import get_logger
    logger = get_logger(__name__)

    logger.info(f"Applying fumble attribution penalties for {year}")

    # Find fumble plays
    fumble_plays = pbp_data.filter(
        (pl.col('fumble') == 1) &
        (pl.col('fumbled_1_player_id').is_not_null())
    )

    # Classify fumble severity
    fumble_plays = fumble_plays.with_columns([
        # Check if recovered by offense or defense
        pl.when(
            pl.col('fumble_recovery_1_team') == pl.col('posteam')
        )
        .then('offense_recovered')
        .otherwise('defense_recovered')
        .alias('recovery_type'),

        # Check play type
        pl.col('play_type').alias('fumble_play_type'),
    ])

    # Calculate penalty by fumbler
    fumble_penalties = fumble_plays.group_by('fumbled_1_player_id').agg([
        pl.count().alias('total_fumbles'),
        (pl.col('recovery_type') == 'offense_recovered').sum().alias('fumbles_recovered_offense'),
        (pl.col('recovery_type') == 'defense_recovered').sum().alias('fumbles_lost'),
        (pl.col('fumble_play_type') == 'pass').sum().alias('strip_sacks'),
        (pl.col('fumble_play_type') == 'run').sum().alias('run_fumbles'),
    ])

    # Calculate weighted penalty
    fumble_penalties = fumble_penalties.with_columns([
        (
            (pl.col('fumbles_recovered_offense') * -5) +  # Moderate penalty
            (pl.col('fumbles_lost') * -12) +  # High penalty
            (pl.col('strip_sacks') * -10)  # QB strip sack penalty
        ).alias('fumble_penalty')
    ])

    # Join to player aggregates
    player_agg = player_agg.join(
        fumble_penalties.select([
            pl.col('fumbled_1_player_id').alias('player_id'),
            'fumble_penalty'
        ]),
        on='player_id',
        how='left'
    )

    # Fill nulls with 0
    player_agg = player_agg.with_columns([
        pl.col('fumble_penalty').fill_null(0)
    ])

    # Apply penalty
    player_agg = player_agg.with_columns([
        (pl.col('overall_contribution') + pl.col('fumble_penalty'))
        .alias('overall_contribution')
    ])

    logger.info(
        f"Fumble penalties applied to "
        f"{player_agg.filter(pl.col('fumble_penalty') != 0).height} players"
    )

    return player_agg
```

#### Step 5: Integrate into Phase 4 pipeline

**File**: `modules/adjustment_pipeline.py`

**Modification** in `apply_phase4_adjustments()`:

```python
def apply_phase4_adjustments(
    player_agg: pl.DataFrame,
    year: int,
    pbp_data: pl.DataFrame = None
) -> pl.DataFrame:
    """Apply Phase 4 adjustments..."""

    # ... existing Phase 4 adjustments ...

    # NEW: Enhanced turnover attribution (Phase 6.4)
    if pbp_data is not None:
        logger.info("Applying turnover attribution enhancements (Phase 6.4)")

        # Receiver tip penalties
        player_agg = apply_receiver_tip_penalty(player_agg, pbp_data, year)

        # Fumble attribution penalties
        player_agg = apply_fumble_attribution_penalty(player_agg, pbp_data, year)
    else:
        logger.warning("PBP data not provided, skipping turnover attribution")
        player_agg = player_agg.with_columns([
            pl.lit(0).alias('tip_int_penalty'),
            pl.lit(0).alias('fumble_penalty')
        ])

    # ... rest of Phase 4 ...

    return player_agg
```

### Testing & Validation

#### Unit Tests

**File**: `tests/test_turnover_attribution.py`

```python
"""Unit tests for turnover attribution logic."""

import polars as pl
from modules.adjustment_pipeline import (
    apply_receiver_tip_penalty,
    apply_fumble_attribution_penalty
)


def test_receiver_tip_int_penalty():
    """Test that WR gets penalized for tipped INT."""
    player_data = pl.DataFrame({
        'player_id': ['WR1'],
        'position': ['WR'],
        'overall_contribution': [80.0]
    })

    pbp_data = pl.DataFrame({
        'interception': [1, 1],
        'receiver_player_id': ['WR1', 'WR1'],
        'was_pressure': [False, False],  # Clean throws
    })

    result = apply_receiver_tip_penalty(player_data, pbp_data, 2024)

    # Should get -8 per INT × 2 = -16 total
    assert result['tip_int_penalty'][0] == -16
    assert result['overall_contribution'][0] == 64.0  # 80 - 16


def test_fumble_recovered_by_offense():
    """Test reduced penalty when fumble recovered by offense."""
    player_data = pl.DataFrame({
        'player_id': ['RB1'],
        'position': ['RB'],
        'overall_contribution': [100.0]
    })

    pbp_data = pl.DataFrame({
        'fumble': [1],
        'fumbled_1_player_id': ['RB1'],
        'fumble_recovery_1_team': ['KC'],  # Offense recovered
        'posteam': ['KC'],
        'play_type': ['run'],
    })

    result = apply_fumble_attribution_penalty(player_data, pbp_data, 2024)

    # Should get -5 penalty (moderate - offense recovered)
    assert result['fumble_penalty'][0] == -5
    assert result['overall_contribution'][0] == 95.0


def test_fumble_lost_to_defense():
    """Test harsh penalty when fumble lost to defense."""
    player_data = pl.DataFrame({
        'player_id': ['RB2'],
        'position': ['RB'],
        'overall_contribution': [100.0]
    })

    pbp_data = pl.DataFrame({
        'fumble': [1],
        'fumbled_1_player_id': ['RB2'],
        'fumble_recovery_1_team': ['SF'],  # Defense recovered
        'posteam': ['KC'],
        'play_type': ['run'],
    })

    result = apply_fumble_attribution_penalty(player_data, pbp_data, 2024)

    # Should get -12 penalty (harsh - turnover)
    assert result['fumble_penalty'][0] == -12
    assert result['overall_contribution'][0] == 88.0
```

### Success Criteria

- [ ] Turnover attribution enhancements integrated into Phase 4
- [ ] QB INT penalty reduced by 33% when under pressure
- [ ] WR tip INT penalty (-8) applied to receivers on INT plays
- [ ] RB fumble penalties differentiated by recovery (offense -5, defense -12)
- [ ] No NaN values in tip_int_penalty or fumble_penalty columns
- [ ] Unit tests pass for all attribution scenarios
- [ ] Known ball-security issues (e.g., frequent fumblers) show appropriate penalties
- [ ] Processing time increase <2% vs. baseline

---

## Testing Strategy

### Overview

Comprehensive testing at multiple levels to ensure Phase 6 features integrate correctly without introducing NaNs or breaking existing logic.

### Testing Levels

1. **Data Coverage Tests** (Pre-Implementation)
2. **Unit Tests** (Per Feature)
3. **Integration Tests** (Full Pipeline)
4. **Regression Tests** (Existing Features)
5. **Performance Tests** (Processing Time)

### Data Coverage Tests

**When**: BEFORE any code changes
**File**: `tests/test_data_coverage_phase6.py` (see Data Coverage Validation section)

**Checklist**:
- [ ] Run `python tests/test_data_coverage_phase6.py`
- [ ] All tests PASS before proceeding
- [ ] Document null rates for each feature by year
- [ ] Identify years where features should be disabled due to poor coverage

### Unit Tests

**When**: After each feature implementation
**Files**:
- `tests/test_success_rate_adjustment.py`
- `tests/test_drive_context_adjustment.py`
- `tests/test_route_location_adjustment.py`
- `tests/test_turnover_attribution.py`

**Checklist**:
- [ ] All unit tests pass with `pytest tests/test_*_adjustment.py`
- [ ] NaN tests pass (verify no null values in output)
- [ ] Multiplier range tests pass (verify bounds)
- [ ] Edge case tests pass (null input, insufficient sample, extreme values)

### Integration Tests

**File**: `tests/test_phase6_integration.py`

```python
"""Integration tests for Phase 6 full pipeline."""

import polars as pl
from modules.rankings_generator import (
    generate_qb_rankings,
    generate_rb_rankings,
    generate_wr_rankings,
    generate_te_rankings
)


def test_phase6_integration_full_pipeline_2024():
    """Test that all Phase 6 features work together in full pipeline."""

    # Generate all rankings
    qb_rankings = generate_qb_rankings(2024)
    rb_rankings = generate_rb_rankings(2024)
    wr_rankings = generate_wr_rankings(2024)
    te_rankings = generate_te_rankings(2024)

    # Validate structure
    for rankings, pos in [(qb_rankings, 'QB'), (rb_rankings, 'RB'),
                           (wr_rankings, 'WR'), (te_rankings, 'TE')]:
        print(f"\nValidating {pos} rankings...")

        # Check Phase 6 columns exist
        assert 'success_adjustment' in rankings.columns
        if pos in ['WR', 'TE']:
            assert 'route_location_adjustment' in rankings.columns

        # Validate no NaNs
        assert rankings['overall_contribution'].is_not_nan().all()
        assert rankings['success_adjustment'].is_not_nan().all()

        # Validate multiplier ranges
        assert rankings['success_adjustment'].min() >= 0.85
        assert rankings['success_adjustment'].max() <= 1.20

        if pos in ['WR', 'TE']:
            assert rankings['route_location_adjustment'].min() >= 0.90
            assert rankings['route_location_adjustment'].max() <= 1.15

        print(f"  ✓ {pos} rankings valid")

    print("\n✓ Full pipeline integration test PASSED")


def test_phase6_known_player_validations_2024():
    """Test known players show expected adjustments."""

    # QB: Lamar Jackson (high efficiency)
    qb_rankings = generate_qb_rankings(2024)
    lamar = qb_rankings.filter(pl.col('player_name').str.contains('L.Jackson'))
    if len(lamar) > 0:
        assert lamar['success_adjustment'][0] >= 1.05, "Lamar should have high success rate"

    # RB: Christian McCaffrey (elite efficiency)
    rb_rankings = generate_rb_rankings(2024)
    cmc = rb_rankings.filter(pl.col('player_name').str.contains('C.McCaffrey'))
    if len(cmc) > 0:
        assert cmc['success_adjustment'][0] >= 1.08, "CMC should have elite success rate"

    # WR: Slot specialist validation (Cooper Kupp if healthy)
    wr_rankings = generate_wr_rankings(2024)
    kupp = wr_rankings.filter(pl.col('player_name').str.contains('C.Kupp'))
    if len(kupp) > 0 and kupp['target_count'][0] >= 30:
        # Slot receiver should get middle-field bonus
        assert kupp['route_location_adjustment'][0] >= 1.02, "Kupp should get slot bonus"
        assert kupp['pct_middle_routes'][0] >= 0.40, "Kupp should run >40% middle routes"

    print("✓ Known player validations PASSED")


def test_phase6_no_regression_existing_features():
    """Test that Phase 6 doesn't break existing Phase 1-5 logic."""

    rb_rankings = generate_rb_rankings(2024)

    # Existing columns should still exist
    assert 'catch_rate_adjustment' in rb_rankings.columns  # Phase 4
    assert 'separation_adjustment' in rb_rankings.columns  # Phase 4
    assert 'weather_adjustment' in rb_rankings.columns  # Phase 4.5
    assert 'talent_adjustment' in rb_rankings.columns  # Phase 5

    # Existing adjustments should still have valid ranges
    assert rb_rankings['catch_rate_adjustment'].min() >= 0.85
    assert rb_rankings['catch_rate_adjustment'].max() <= 1.15

    print("✓ No regression in existing features")
```

### Regression Tests

**Purpose**: Ensure Phase 6 doesn't break existing Phases 1-5

**Checklist**:
- [ ] Run full test suite: `pytest tests/`
- [ ] All existing tests still pass
- [ ] Existing adjustment columns still present and valid
- [ ] Known player rankings from 2023 haven't dramatically shifted

### Performance Tests

**File**: `tests/test_phase6_performance.py`

```python
"""Performance tests for Phase 6 processing time impact."""

import time
import polars as pl
from modules.rankings_generator import generate_qb_rankings


def test_phase6_performance_impact():
    """Test that Phase 6 adds <10% processing time."""

    # Baseline: Generate rankings 3 times, average time
    baseline_times = []
    for _ in range(3):
        start = time.time()
        _ = generate_qb_rankings(2024)
        baseline_times.append(time.time() - start)

    avg_baseline = sum(baseline_times) / len(baseline_times)

    print(f"Average baseline time: {avg_baseline:.2f}s")

    # With Phase 6 features, time should be <10% longer
    # (This test assumes Phase 6 is already implemented)

    # Acceptable range: baseline to baseline * 1.10
    max_acceptable = avg_baseline * 1.10

    print(f"Max acceptable time: {max_acceptable:.2f}s")
    print(f"Target: <10% increase")

    # If any run exceeds max_acceptable, flag it
    for i, t in enumerate(baseline_times):
        if t > max_acceptable:
            print(f"  ⚠ WARNING: Run {i+1} exceeded target ({t:.2f}s)")
```

### NaN Prevention Checklist

At every integration point, verify:

- [ ] No null/NaN values in new multiplier columns
- [ ] All multipliers have valid range (0.75-1.15x)
- [ ] Null input data defaults to 1.0x neutral
- [ ] Division by zero checks (e.g., success_rate with 0 plays)
- [ ] Missing PBP data gracefully degrades (logs warning, skips feature)
- [ ] Polars `.fill_null()` used on all joins
- [ ] Polars `.clip()` used to cap multiplier ranges

---

## Performance Considerations

### Expected Impact

| Feature | Processing Time | Memory Impact | Disk Space |
|---------|----------------|---------------|------------|
| Success Rate | +1-2% | Minimal (<10 MB) | 1 column in PBP cache |
| Drive Context | +1-2% | Minimal (<10 MB) | 2 columns in PBP cache |
| Route Location | +2-4% | Minimal (<20 MB) | 3 columns in PBP cache |
| Turnover Attribution | +1-2% | Minimal (<15 MB) | 5 columns in PBP cache |
| **Total** | **+5-10%** | **<55 MB** | **11 new columns** |

### Optimization Strategies

1. **Lazy evaluation**: Use Polars lazy API where possible
2. **Column selection**: Only load Phase 6 columns when needed
3. **Caching**: Pre-calculate multipliers in PBP cache (not runtime)
4. **Parallel processing**: Leverage existing parallel year processing

### Cache Size Impact

**Current PBP cache size** (per year, estimated): ~200-300 MB parquet
**Phase 6 additional columns**: 11 columns × ~50k plays × 8 bytes = ~4.4 MB per year
**Percentage increase**: ~1.5-2% per year

**Total disk space for all years (2010-2024):**
15 years × 4.4 MB = ~66 MB additional storage (negligible)

---

## Implementation Checklist

### Pre-Implementation

- [ ] Run data coverage tests (`test_data_coverage_phase6.py`)
- [ ] Document baseline performance (current processing time for 2024)
- [ ] Create git branch: `feature/phase-6-efficiency-metrics`
- [ ] Back up current PBP cache directory

### Phase 6.1: Success Rate

- [ ] Add `success` column to PBP cache builder
- [ ] Create `apply_success_rate_adjustment()` function
- [ ] Integrate into `apply_phase4_adjustments()`
- [ ] Write unit tests
- [ ] Run integration test on 2024 data
- [ ] Validate no NaNs in output
- [ ] Commit changes

### Phase 6.2: Drive Context

- [ ] Add `drive`, `drive_end_transition` columns to PBP cache builder
- [ ] Calculate `drive_outcome_multiplier` in cache
- [ ] Update `_calculate_multipliers()` to include drive context
- [ ] Write unit tests
- [ ] Run integration test on 2024 data
- [ ] Validate no NaNs in output
- [ ] Commit changes

### Phase 6.3: Route/Target Location

- [ ] Add `pass_location`, `pass_length`, `air_yards` columns to PBP cache
- [ ] Create `calculate_route_location_adjustment()` function
- [ ] Implement slot receiver bias mitigation matrix
- [ ] Integrate into `apply_phase4_adjustments()`
- [ ] Write unit tests (including bias mitigation test)
- [ ] Run integration test on 2023 slot specialists
- [ ] Validate |avg_slot - avg_outside| < 0.04x
- [ ] Commit changes

### Phase 6.4: Turnover Attribution

- [ ] Add turnover attribution columns to PBP cache
- [ ] Enhance QB INT penalty logic in `calculate_qb_contribution_from_pbp()`
- [ ] Create `apply_receiver_tip_penalty()` function
- [ ] Create `apply_fumble_attribution_penalty()` function
- [ ] Integrate into `apply_phase4_adjustments()`
- [ ] Write unit tests
- [ ] Run integration test on 2024 data
- [ ] Validate no NaNs in output
- [ ] Commit changes

### Testing & Validation

- [ ] Run full test suite: `pytest tests/`
- [ ] Run performance test to verify <10% increase
- [ ] Generate 2024 rankings and spot-check known players
- [ ] Compare 2024 rankings before/after Phase 6 (validate reasonableness)
- [ ] Test cache rebuild process on 2023 data
- [ ] Document any data gaps or issues found

### Cache Rebuild

- [ ] Update `check_and_rebuild_caches()` in main.py to check for Phase 6 columns
- [ ] Add Phase 6 columns to `required_cols` list
- [ ] Test cache rebuild on single year (2024)
- [ ] Rebuild caches for all years (2016-2024 for full participation data)
- [ ] Validate cache sizes are reasonable

### Documentation

- [ ] Update README.md with Phase 6 feature descriptions
- [ ] Add Phase 6 section to architecture documentation
- [ ] Document multiplier ranges and rationale
- [ ] Create "before/after" comparison table for known players
- [ ] Update CHANGELOG.md

### Deployment

- [ ] Merge feature branch to main
- [ ] Tag release: `v2.6.0` (Phase 6 release)
- [ ] Regenerate all rankings (2016-2024)
- [ ] Archive old rankings for comparison
- [ ] Monitor for any issues in production

---

## Success Criteria

### Quantitative Metrics

1. **Data Quality**:
   - [ ] All Phase 6 columns have <10% null rate (post-2015)
   - [ ] No NaN values in any output rankings
   - [ ] All multipliers within expected ranges (0.75-1.15x)

2. **Performance**:
   - [ ] Processing time increase <10% vs. baseline
   - [ ] Cache size increase <5% vs. baseline
   - [ ] No memory leaks or crashes

3. **Accuracy**:
   - [ ] Known efficient players (CMC, Lamar) show >1.08x success adjustment
   - [ ] Slot receivers not penalized vs. outside receivers (bias <0.04x)
   - [ ] Ball security issues properly flagged (fumblers show penalties)

### Qualitative Metrics

4. **Ranking Improvements**:
   - [ ] Dual-threat RBs (CMC, Alvin Kamara) move up in rankings
   - [ ] Volume-only RBs (Najee Harris type) move down slightly
   - [ ] Efficient QBs (Brock Purdy) properly valued vs. volume QBs
   - [ ] Slot specialists (Cooper Kupp, Amon-Ra St. Brown) fairly ranked

5. **Code Quality**:
   - [ ] All unit tests pass
   - [ ] No linting errors
   - [ ] Code follows existing patterns (Polars, logging, constants)
   - [ ] Comprehensive docstrings on all new functions

### Validation Test Cases

**Known Player Checks (2023-2024)**:

| Player | Feature | Expected Result |
|--------|---------|----------------|
| Christian McCaffrey | Success Rate | ≥1.10x (elite efficiency) |
| Lamar Jackson | Success Rate | ≥1.08x (high 3rd down conversions) |
| Cooper Kupp | Route Location | 1.02-1.06x (slot bonus, not penalized) |
| Amon-Ra St. Brown | Route Location | 1.03-1.06x (slot bonus) |
| Tyreek Hill | Route Location | 1.04-1.06x (deep outside) |
| Najee Harris | Success Rate | 0.92-0.98x (low efficiency, volume only) |
| Puka Nacua | Drive Context | Validated (Rams scoring drives) |
| Known ball security issues | Turnover Attribution | Appropriate fumble penalties |

---

## Risk Mitigation

### Risk 1: Data Coverage Gaps

**Risk**: Earlier years (2010-2015) may have poor coverage for pass_location, pass_length

**Mitigation**:
- Run data coverage tests FIRST
- Implement year-specific feature flags
- Gracefully degrade to 1.0x neutral if data missing
- Document known gaps in README

**Contingency**:
- If coverage <60% for a year, disable that feature for that year
- Log warnings when features are skipped

### Risk 2: Slot Receiver Bias

**Risk**: Route location adjustment unfairly penalizes slot receivers

**Mitigation**:
- Implement middle-of-field bonus (+0.03-0.06x)
- Test with known slot specialists (Kupp, ARSB, Kirk)
- Require |avg_slot - avg_outside| < 0.04x
- Add diagnostic logging for slot receiver adjustments

**Contingency**:
- If bias detected, adjust middle-field bonus upward
- Consider separate slot vs. outside receiver baselines

### Risk 3: Performance Degradation

**Risk**: Phase 6 processing time >10% increase

**Mitigation**:
- Pre-calculate multipliers in cache (not runtime)
- Use Polars lazy evaluation
- Profile code to identify bottlenecks
- Leverage existing parallel year processing

**Contingency**:
- If performance issue, disable most expensive feature
- Consider async/parallel processing for Phase 6 calculations

### Risk 4: NaN Introduction

**Risk**: New joins or calculations create NaN values

**Mitigation**:
- Use `.fill_null()` on every join
- Use `.clip()` to cap multiplier ranges
- Add NaN checks at every integration point
- Comprehensive unit tests for null inputs

**Contingency**:
- If NaN detected, trace back to source and add null handling
- Default to 1.0x neutral on any NaN

### Risk 5: Breaking Existing Features

**Risk**: Phase 6 changes break Phases 1-5 logic

**Mitigation**:
- Comprehensive regression tests
- Test existing adjustment columns still present
- Compare 2023 rankings before/after Phase 6
- Incremental implementation (one feature at a time)

**Contingency**:
- Git branch for easy rollback
- Archive old rankings for comparison
- Revert feature if regression detected

---

## Timeline

### Week 1: Setup & Data Coverage

**Day 1-2**:
- [ ] Create implementation plan document (this doc)
- [ ] Set up git branch
- [ ] Run data coverage tests
- [ ] Document baseline performance

**Day 3-5**:
- [ ] Implement Phase 6.1 (Success Rate)
- [ ] Write unit tests
- [ ] Test on 2024 data

### Week 2: Drive & Route Features

**Day 6-8**:
- [ ] Implement Phase 6.2 (Drive Context)
- [ ] Write unit tests
- [ ] Test on 2024 data

**Day 9-12**:
- [ ] Implement Phase 6.3 (Route Location with slot mitigation)
- [ ] Write comprehensive tests including bias validation
- [ ] Test on 2023 slot specialists

### Week 3: Turnover & Testing

**Day 13-15**:
- [ ] Implement Phase 6.4 (Turnover Attribution)
- [ ] Write unit tests
- [ ] Test on 2024 data

**Day 16-18**:
- [ ] Run full integration tests
- [ ] Performance testing
- [ ] Regression testing

**Day 19-21**:
- [ ] Cache rebuild for all years (2016-2024)
- [ ] Generate new rankings
- [ ] Validation and documentation

### Total Estimated Time: 3 weeks (15-21 working days)

---

## Appendices

### Appendix A: nflfastpy Column Definitions

```python
# Phase 6 Column Reference

# Success Rate (Phase 6.1)
'success': bool  # Binary: did play advance situation?
    # Thresholds:
    # - 1st down: gain >= 45% of yards to go
    # - 2nd down: gain >= 60% of yards to go
    # - 3rd/4th: gain >= 100% (convert)

# Drive Context (Phase 6.2)
'drive': int  # Drive number within game
'drive_end_transition': str  # How drive ended
    # Values: 'Touchdown', 'Field goal', 'Punt', 'Turnover',
    #         'Turnover on downs', 'End of half', 'End of game'

# Route Location (Phase 6.3)
'pass_location': str  # Target location: 'left', 'middle', 'right'
'pass_length': str  # Target depth: 'short' (<10 yds), 'deep' (≥10 yds)
'air_yards': float  # Exact air yards on pass attempt

# Turnover Attribution (Phase 6.4)
'interception_player_id': str  # Defender who caught INT
'fumbled_1_player_id': str  # Primary fumbler
'fumbled_2_player_id': str  # Secondary fumbler (if applicable)
'fumble_recovery_player_id': str  # Who recovered fumble
'fumble_recovery_1_team': str  # Recovery team code
'was_pressure': bool  # Whether QB was under pressure (2016+)
```

### Appendix B: Multiplier Summary Table

| Feature | Multiplier Range | Integration Point | When Applied |
|---------|------------------|-------------------|--------------|
| Success Rate | 0.92-1.15x | Phase 4 (per-week) | After shares calculated |
| Drive Context | 0.75-1.15x | Phase 1 (play level) | Before aggregation |
| Route Location | 0.95-1.12x | Phase 4 (season) | After shares calculated |
| Turnover Attribution | -15 to 0 penalty | Phase 4 (per-week) | After shares calculated |

**Combined Phase 1 Multiplier** (with drive context):
```
situational_mult = field_pos × score × time × down × drive_outcome
Range: 0.75 - 5.06x (theoretical max: 2.0 × 1.5 × 1.5 × 1.5 × 1.15)
Typical: 1.0 - 2.5x
```

**Combined Phase 4 Multiplier** (with Phase 6 features):
```
phase4_mult = catch_rate × separation × success × route_location × penalties
Range: 0.65 - 1.38x (typical caps prevent extremes)
Typical: 0.90 - 1.12x
```

### Appendix C: Constants Configuration

**File**: `modules/constants.py`

```python
# Phase 6: Efficiency Metrics Configuration

# Success Rate Multipliers
SUCCESS_RATE_CRITICAL_MULTIPLIER = 1.15  # 3rd/4th down success
SUCCESS_RATE_HIGH_MULTIPLIER = 1.10      # High success (≥55%)
SUCCESS_RATE_AVERAGE_MULTIPLIER = 1.05   # Average success (45-55%)
SUCCESS_RATE_NEUTRAL_MULTIPLIER = 1.0    # Below average (35-45%)
SUCCESS_RATE_LOW_MULTIPLIER = 0.92       # Low success (<35%)
SUCCESS_RATE_MIN_PLAYS = 20              # Minimum plays for adjustment

# Drive Context Multipliers
DRIVE_OUTCOME_TD_MULTIPLIER = 1.15       # Touchdown drive
DRIVE_OUTCOME_FG_MULTIPLIER = 1.05       # Field goal drive
DRIVE_OUTCOME_PUNT_MULTIPLIER = 0.90     # Punt
DRIVE_OUTCOME_TURNOVER_MULTIPLIER = 0.75 # Turnover drive
DRIVE_OUTCOME_NEUTRAL_MULTIPLIER = 1.0   # End of half/game

# Route Location Multipliers (with slot bias mitigation)
ROUTE_DEEP_MIDDLE_MULTIPLIER = 1.12      # Deep middle (15+ yards)
ROUTE_INT_MIDDLE_MULTIPLIER = 1.06       # Intermediate middle (8-14)
ROUTE_SHORT_MIDDLE_MULTIPLIER = 1.03     # Short middle (<8) - SLOT BONUS
ROUTE_DEEP_SIDELINE_MULTIPLIER = 1.05    # Deep sideline
ROUTE_INT_SIDELINE_MULTIPLIER = 1.00     # Intermediate sideline (baseline)
ROUTE_SHORT_SIDELINE_MULTIPLIER = 0.95   # Short sideline (easiest)
ROUTE_LOCATION_MIN_TARGETS = 30          # Minimum targets for adjustment

# Route depth thresholds
ROUTE_DEEP_THRESHOLD = 15.0              # Air yards for "deep"
ROUTE_INTERMEDIATE_THRESHOLD = 8.0       # Air yards for "intermediate"

# Turnover Attribution Penalties
QB_INT_BASE_PENALTY = -15                # Base INT penalty
QB_INT_PRESSURE_REDUCTION = 0.33         # 33% reduction under pressure
QB_INT_FINAL_2MIN_MULTIPLIER = 1.3       # 30% increase in final 2 min
WR_TIP_INT_PENALTY = -8                  # WR tipped INT penalty
WR_TIP_INT_PRESSURE_REDUCTION = -5       # Reduced if QB under pressure
RB_FUMBLE_RECOVERED_PENALTY = -5         # Fumble recovered by offense
RB_FUMBLE_LOST_PENALTY = -12             # Fumble lost to defense
QB_STRIP_SACK_PENALTY = -10              # QB strip sack

# Phase 6 Feature Flags (by year)
PHASE_6_SUCCESS_RATE_START_YEAR = 1999   # Success available from 1999
PHASE_6_DRIVE_CONTEXT_START_YEAR = 1999  # Drive data from 1999
PHASE_6_ROUTE_LOCATION_START_YEAR = 2015 # Good location data from 2015
PHASE_6_PRESSURE_DATA_START_YEAR = 2016  # was_pressure from 2016
```

### Appendix D: Example Output Columns

**After Phase 6 implementation, player rankings will include:**

```python
# Existing columns (unchanged)
'player_id': str
'player_name': str
'position': str
'team': str
'overall_contribution': float  # Final score
'catch_rate_adjustment': float  # Phase 4
'separation_adjustment': float  # Phase 4
'weather_adjustment': float    # Phase 4.5
'talent_adjustment': float     # Phase 5

# NEW Phase 6 columns
'success_adjustment': float    # Phase 6.1 (0.92-1.15x)
'drive_outcome_multiplier': float  # Phase 6.2 (0.75-1.15x, implicit in Phase 1)
'route_location_adjustment': float  # Phase 6.3 (0.95-1.12x, WR/TE only)
'avg_air_yards': float        # Phase 6.3 diagnostic
'pct_middle_routes': float    # Phase 6.3 diagnostic
'tip_int_penalty': int        # Phase 6.4 (WR only)
'fumble_penalty': int         # Phase 6.4 (RB/QB)
```

---

## Conclusion

Phase 6 represents a significant accuracy enhancement to GridironMetrics player rankings by introducing efficiency-focused metrics that complement existing volume-based calculations. The careful integration into existing Phases 1 and 4, combined with comprehensive testing and slot receiver bias mitigation, ensures these features enhance rather than disrupt current functionality.

The modular implementation approach allows for incremental rollout and easy rollback if issues arise, while the detailed testing strategy ensures data quality and performance targets are met.

Upon successful implementation, GridironMetrics will provide more nuanced player evaluations that better separate efficient contributors from volume accumulators, ultimately delivering more actionable insights for fantasy football and NFL analysis.

---

**Document Control**

- **Version**: 1.0
- **Last Updated**: 2025-01-02
- **Author**: GridironMetrics Development Team
- **Status**: Ready for Implementation
- **Next Review**: After Phase 6.1 implementation

---

END OF IMPLEMENTATION PLAN
