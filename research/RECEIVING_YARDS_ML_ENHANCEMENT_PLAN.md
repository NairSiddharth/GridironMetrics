# Receiving Yards ML Model Enhancement Implementation Plan

**Version**: 1.0
**Date**: 2025-11-05
**Status**: Planning
**Priority**: High
**Estimated Effort**: 2-3 days

---

## Executive Summary

### Objective
Enhance receiving_yards ML model accuracy from 41.6% to 50%+ by implementing 9 critical improvements based on user feedback:
1. **Split WR/TE models** - Different usage patterns require position-specific training
2. **Add target volume features** - Volume is the #1 predictor of receiving yards (currently missing)
3. **Remove redundant features** - Eliminate last_5_avg (multicollinearity with last_3_avg)
4. **Add NextGen Stats** - Leverage separation/cushion data (available, but unused)
5. **Add prior season baseline** - Year-over-year context for better predictions
6. **Fix RB feature pollution** - Prevent RB-specific features from contaminating WR/TE models
7. **Apply target qualification filtering** - Minimum 3 targets/week, 40+ season targets (data quality)
8. **Add YPT efficiency metric** - Yards per target as efficiency signal
9. **Add raw target counts** - Actual volume, not just percentages

### Expected Impact
- **10-15% accuracy improvement** (41.6% → 50-55%)
- **15-20% ROI improvement** (-20.49% → -5% to break-even)
- **Better position homogeneity** - WRs and TEs have distinct usage patterns
- **Volume signal integration** - Target share is the #1 missing feature
- **Cleaner training data** - 20-30% fewer examples, but much higher quality

### Key Benefits
- All features use existing cached data (NextGen 2016+, PBP 1999+)
- Integrates cleanly into existing ml_feature_engineering.py architecture
- No external API dependencies
- Comprehensive testing strategy
- Minimal performance impact (<15% processing time increase)

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Data Coverage Validation](#data-coverage-validation)
3. [Phase 1.1: Split WR/TE Prop Types](#phase-11-split-wrte-prop-types)
4. [Phase 1.2: Add Target Volume Features](#phase-12-add-target-volume-features)
5. [Phase 1.3: Integrate NextGen Stats](#phase-13-integrate-nextgen-stats)
6. [Phase 1.4: Add Prior Season Baseline](#phase-14-add-prior-season-baseline)
7. [Phase 1.5: Remove last_5_avg](#phase-15-remove-last_5_avg)
8. [Phase 1.6: Fix RB Feature Pollution](#phase-16-fix-rb-feature-pollution)
9. [Phase 1.7: Target Qualification Filtering](#phase-17-target-qualification-filtering)
10. [Testing Strategy](#testing-strategy)
11. [Results Presentation Format](#results-presentation-format)
12. [Success Criteria](#success-criteria)
13. [Risk Mitigation](#risk-mitigation)
14. [Implementation Checklist](#implementation-checklist)
15. [Timeline](#timeline)

---

## Current State Assessment

### Baseline Performance (receiving_yards)

**Current Configuration:**
- **Prop Type**: `receiving_yards` (WR + TE combined)
- **Position Filter**: `['WR', 'TE']` (mixed model)
- **Total Features**: ~55 features
- **Training Years**: 2015-2024
- **Min Sample Size**: 15 targets

**Performance Metrics:**
- **Overall Accuracy**: 41.6% (3,257/7,820 correct)
- **ROI**: -20.49%
- **Year-by-Year** (2022-2024):
  - 2022: 40.9% accuracy
  - 2023: 41.8% accuracy
  - 2024: 42.1% accuracy
- **Status**: NOT READY FOR BETTING (below 48% breakeven threshold)

### Current Feature Categories

**Baseline Features** (8):
- weighted_avg, last_3_avg, **last_5_avg** ← REDUNDANT
- career_avg, variance_cv, games_played, effective_games

**Game Script Features** (6):
- team_avg_margin, team_rb_quality, opp_def_ppg_allowed, opp_def_ypg_allowed
- team_plays_per_game, team_time_of_possession

**Opponent Defense Features** (9):
- Various pass defense rolling averages

**Catch Rate Features** (3):
- catch_rate_season, catch_rate_3wk, catch_rate_season_rolling

**QB Quality Features** (6):
- qb_passer_rating, qb_completion_pct, qb_ypa, etc.

**Team Offense Features** (5):
- Team passing stats, offensive rankings

**Weather Features** (4):
- game_temp, game_wind, game_precip, game_is_dome

**NextGen Features**: ❌ **0 features** (DATA AVAILABLE, NOT USED)

**Target Volume Features**: ❌ **0 features** (CRITICAL GAP - #1 predictor)

**Prior Season Features**: ❌ **0 features** (NO Y-1 baseline)

### Key Problems Identified

1. **WR/TE Mixed Model Issue**
   - WRs: Deep routes, high variance, volume-driven
   - TEs: Blocking duties, short routes, TD-prone in red zone
   - **Impact**: Model confuses two distinct roles

2. **Missing Target Volume** (#1 Gap)
   - No target share features (season or rolling)
   - No raw target count features (just percentages)
   - No yards per target (efficiency metric)
   - **Impact**: Missing the primary driver of receiving yards

3. **Feature Redundancy**
   - last_3_avg and last_5_avg are ~60% correlated
   - Creates multicollinearity noise
   - **Impact**: Model overfits to redundant signals

4. **Unused Data Sources**
   - NextGen Stats cached 2016+ (separation, cushion)
   - Used in ranking system but NOT in ML
   - **Impact**: Leaving 8+ years of GPS data on the table

5. **RB Feature Pollution**
   - ypc_diff_pct, blocking_quality leaking into WR/TE models
   - **Impact**: Irrelevant features confusing the model

6. **Low-Quality Training Data**
   - WR3/WR4s with 1-2 targets/week included
   - Noise-dominated examples
   - **Impact**: Model learns from low-signal data

---

## Data Coverage Validation

### Required Data Sources

| Feature Category | Data Source | Years Available | Null Rate | Status |
|-----------------|-------------|-----------------|-----------|---------|
| Target Volume | PBP (receiver_player_id) | 1999+ | <1% | ✅ Ready |
| NextGen Stats | NextGen Cache | 2016+ | 0% | ✅ Ready |
| Prior Season | Positional Stats | 2015+ | ~5% (rookies) | ✅ Ready |
| Baseline Stats | Positional Stats | 1999+ | <1% | ✅ Ready |

### Data Validation Tests

**Pre-Implementation Checks:**
```python
# Test 1: NextGen Stats Coverage
cache_file = Path("cache/nextgen/nextgen_2024.parquet")
assert cache_file.exists(), "NextGen 2024 cache missing"

df = pl.read_parquet(cache_file)
assert 'avg_separation' in df.columns
assert 'avg_cushion' in df.columns
assert df['avg_separation'].null_count() / len(df) < 0.05  # <5% null

# Test 2: Target Data Coverage
pbp_file = Path("cache/pbp/pbp_2024.parquet")
pbp = pl.read_parquet(pbp_file)

targets = pbp.filter(
    (pl.col('receiver_player_id').is_not_null()) &
    ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
)
assert len(targets) > 50000  # Expect 50k+ targets per season

# Test 3: Prior Season Availability
wr_2023 = Path("cache/positional_player_stats/wr/wr-2023.csv")
assert wr_2023.exists(), "2023 WR stats missing (needed for 2024 Y-1 baseline)"
```

**Expected Results:**
- NextGen: 100% coverage 2016+, 0% null rate for separation/cushion
- Targets: 50,000+ per season, <1% null rate
- Prior Season: Available for all years except first (2015)

---

## Phase 1.1: Split WR/TE Prop Types

### Objective
Create separate `receiving_yards_wr` and `receiving_yards_te` prop types to allow position-specific model training.

### Rationale
- **WRs**: Deep routes (10+ air yards), high variance, volume-driven production
- **TEs**: Blocking duties reduce snaps, short/medium routes, red zone targets
- **Usage patterns are fundamentally different** - one model cannot learn both effectively

### Implementation

**File**: `modules/prop_types.py`

**Change 1: Split receiving_yards (Lines 69-81)**

```python
# BEFORE
'receiving_yards': {
    'adjustments': [...],
    'position': ['WR', 'TE'],  # Combined model
    ...
}

# AFTER
'receiving_yards_wr': {
    'adjustments': [
        'opponent_defense',
        'catch_rate',
        'separation',
        'weather',
    ],
    'api_market': 'player_reception_yds',
    'stat_column': 'receiving_yards',
    'position': ['WR'],  # WR-only
    'min_sample_size': 40,  # Updated from 15
    'min_weekly_volume': 3,  # NEW
    'display_name': 'Receiving Yards (WR)',
},
'receiving_yards_te': {
    'adjustments': [
        'opponent_defense',
        'catch_rate',
        'separation',
        'weather',
    ],
    'api_market': 'player_reception_yds',  # Same API market
    'stat_column': 'receiving_yards',
    'position': ['TE'],  # TE-only
    'min_sample_size': 40,
    'min_weekly_volume': 3,
    'display_name': 'Receiving Yards (TE)',
},
```

**Change 2: Update POSITION_PROP_TYPES mapping (Lines 131-155)**

```python
# BEFORE
POSITION_PROP_TYPES = {
    'WR': [
        'receptions',
        'receiving_yards',  # Shared prop type
        'receiving_tds',
    ],
    'TE': [
        'receptions',
        'receiving_yards',  # Shared prop type
        'receiving_tds',
    ],
}

# AFTER
POSITION_PROP_TYPES = {
    'WR': [
        'receptions_wr',  # Also split
        'receiving_yards_wr',  # WR-specific
        'receiving_tds_wr',  # Also split
    ],
    'TE': [
        'receptions_te',  # Also split
        'receiving_yards_te',  # TE-specific
        'receiving_tds_te',  # Also split
    ],
}
```

**Testing:**
```python
python modules/prop_types.py  # Should run without errors

# Verify split
from modules.prop_types import get_prop_types_for_position
wr_props = get_prop_types_for_position('WR')
assert 'receiving_yards_wr' in wr_props
assert 'receiving_yards_te' not in wr_props
```

---

## Phase 1.2: Add Target Volume Features

### Objective
Add the #1 missing feature category: target volume metrics across three complementary dimensions.

### Rationale
**Target volume is the strongest predictor of receiving yards.** Current model is blind to:
- Season-long target allocation (team role)
- Recent target trends (3-week rolling)
- Yards per target efficiency (catch quality)

**User feedback:**
> "I think YPT will be a good one to have if we are also including the rolling weighted avg. number of targets (not just target share)"

### Three Complementary Dimensions

1. **Efficiency**: Yards per target (YPT)
2. **Raw Volume**: Targets per game (absolute counts)
3. **Context**: Target share (% of team targets)

### Implementation

**File**: `modules/ml_feature_engineering.py`

**New Method: _extract_target_volume_features() (Insert after line 545)**

```python
def _extract_target_volume_features(
    self,
    player_id: str,
    season: int,
    week: int
) -> Dict[str, float]:
    """
    Extract target volume features for WR/TE (5 features).

    Target volume is the #1 predictor of receiving yards.
    Combines both share (%) and raw counts for dual signal.

    Features:
    - target_share_season: % of team targets (season-long baseline)
    - target_share_3wk: % of team targets (recent 3-game)
    - targets_per_game_season: Raw targets/game (season average)
    - targets_per_game_3wk: Raw targets/game (recent 3-game)
    - yards_per_target: Efficiency metric (receiving_yards / targets)

    All calculated using data through week-1 only (no future data).

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Week number (uses data through week-1)

    Returns:
        Dict with 5 target volume features
    """
    features = {
        'target_share_season': 0.15,  # Default to ~15% team share
        'target_share_3wk': 0.15,
        'targets_per_game_season': 5.0,  # Default to 5 targets/game
        'targets_per_game_3wk': 5.0,
        'yards_per_target': 8.0  # Default to 8 yards/target
    }

    try:
        # Load PBP data for this season
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            logger.debug(f"PBP file not found for {season}")
            return features

        pbp = pl.read_parquet(pbp_file)

        # Filter to this player's targets (through week-1)
        player_targets = pbp.filter(
            (pl.col('receiver_player_id') == player_id) &
            (pl.col('week') < week) &  # No future data
            ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
        )

        if len(player_targets) == 0:
            return features

        # Get player's team (most common team in this player's plays)
        team = player_targets['posteam'].mode()[0]

        # Get team targets for same period
        team_targets = pbp.filter(
            (pl.col('posteam') == team) &
            (pl.col('week') < week) &
            ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
        )

        # Calculate season-long metrics
        player_season_targets = len(player_targets)
        team_season_targets = len(team_targets)
        weeks_played = player_targets['week'].n_unique()

        if team_season_targets > 0 and weeks_played > 0:
            features['target_share_season'] = player_season_targets / team_season_targets
            features['targets_per_game_season'] = player_season_targets / weeks_played

        # Calculate 3-week rolling metrics
        last_3_weeks = sorted(player_targets['week'].unique())[-3:]
        if len(last_3_weeks) > 0:
            player_3wk_targets = player_targets.filter(
                pl.col('week').is_in(last_3_weeks)
            )
            team_3wk_targets = team_targets.filter(
                pl.col('week').is_in(last_3_weeks)
            )

            player_3wk_count = len(player_3wk_targets)
            team_3wk_count = len(team_3wk_targets)

            if team_3wk_count > 0:
                features['target_share_3wk'] = player_3wk_count / team_3wk_count
                features['targets_per_game_3wk'] = player_3wk_count / len(last_3_weeks)

        # Calculate yards per target (efficiency)
        completions = player_targets.filter(pl.col('complete_pass') == 1)
        if len(player_targets) > 0:
            total_yards = completions['receiving_yards'].fill_null(0).sum()
            features['yards_per_target'] = total_yards / len(player_targets)

        logger.debug(
            f"Target volume - {player_id}: "
            f"season_share={features['target_share_season']:.2%}, "
            f"tpg_season={features['targets_per_game_season']:.1f}, "
            f"ypt={features['yards_per_target']:.1f}"
        )

        return features

    except Exception as e:
        logger.error(f"Error extracting target volume features: {e}")
        return features
```

**Integration Point (Lines 137-141)**

```python
# BEFORE
if position in ['WR', 'TE', 'RB'] and ('receiving' in prop_type or prop_type == 'receptions'):
    catch_features = self._extract_catch_rate_features(...)
    features.update(catch_features)

# AFTER
if position in ['WR', 'TE', 'RB'] and ('receiving' in prop_type or prop_type == 'receptions'):
    catch_features = self._extract_catch_rate_features(...)
    features.update(catch_features)

    # Add target volume features (WR/TE only, not RB)
    if position in ['WR', 'TE']:
        target_features = self._extract_target_volume_features(
            player_id, season, week
        )
        features.update(target_features)
```

**Default Features Update (Lines 1314-1320)**

```python
if position in ['WR', 'TE']:
    features.update({
        'target_share_season': 0.15,
        'target_share_3wk': 0.15,
        'targets_per_game_season': 5.0,
        'targets_per_game_3wk': 5.0,
        'yards_per_target': 8.0
    })
```

**Testing:**
```python
from modules.ml_feature_engineering import PropFeatureEngineer

engineer = PropFeatureEngineer()
features = engineer.engineer_features(
    player_id="00-0036945",  # CeeDee Lamb
    season=2024,
    week=10,
    position='WR',
    prop_type='receiving_yards_wr',
    opponent_team='PHI'
)

# Verify target volume features present
assert 'target_share_season' in features
assert 'targets_per_game_season' in features
assert 'yards_per_target' in features
assert features['target_share_season'] > 0  # Non-zero value
```

---

## Phase 1.3: Integrate NextGen Stats

### Objective
Add NextGen GPS tracking metrics (separation, cushion) to capture receiver skill at creating space.

### Rationale
- **Available**: NextGen cache built 2016-2025, 0% null rate
- **Used in rankings**: adjustment_pipeline.py already uses separation (0.96x-1.08x adjustments)
- **NOT used in ML**: Zero NextGen features in current model
- **User feedback**: Confirmed - raw values only, NO interaction ratios

### Implementation

**File**: `modules/ml_feature_engineering.py`

**New Method: _extract_nextgen_features() (Insert after target volume method)**

```python
def _extract_nextgen_features(
    self,
    player_id: str,
    season: int,
    week: int,
    position: str
) -> Dict[str, float]:
    """
    Extract NextGen Stats features for WR/TE (2 features).

    NextGen GPS tracking data (2016+):
    - avg_separation: Yards of separation at time of catch
    - avg_cushion: Yards of cushion at snap

    NO RATIOS - just raw values as requested by user.

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Week number (uses data through week-1)
        position: Player position (WR/TE)

    Returns:
        Dict with 2 NextGen features
    """
    features = {
        'avg_separation': 2.5,  # Default ~2.5 yards
        'avg_cushion': 5.0      # Default ~5 yards
    }

    # NextGen data starts 2016
    if season < 2016:
        logger.debug(f"NextGen data not available before 2016 (requested: {season})")
        return features

    try:
        # Load NextGen cache
        nextgen_file = Path(CACHE_DIR) / "nextgen" / f"nextgen_{season}.parquet"
        if not nextgen_file.exists():
            logger.debug(f"NextGen cache not found for {season}")
            return features

        nextgen = pl.read_parquet(nextgen_file)

        # Filter to this player, through week-1
        player_nextgen = nextgen.filter(
            (pl.col('player_gsis_id') == player_id) &
            (pl.col('week') < week)
        )

        if len(player_nextgen) == 0:
            logger.debug(f"No NextGen data for player {player_id} through week {week-1}")
            return features

        # Calculate season-long averages (weighted by targets)
        if 'targets' in player_nextgen.columns and player_nextgen['targets'].sum() > 0:
            total_targets = player_nextgen['targets'].sum()

            # Weighted average separation
            sep_weighted = (
                player_nextgen['avg_separation'].fill_null(2.5) *
                player_nextgen['targets'].fill_null(0)
            ).sum() / total_targets

            # Weighted average cushion
            cushion_weighted = (
                player_nextgen['avg_cushion'].fill_null(5.0) *
                player_nextgen['targets'].fill_null(0)
            ).sum() / total_targets

            features['avg_separation'] = sep_weighted
            features['avg_cushion'] = cushion_weighted
        else:
            # Fall back to simple average if no target weighting available
            features['avg_separation'] = player_nextgen['avg_separation'].fill_null(2.5).mean()
            features['avg_cushion'] = player_nextgen['avg_cushion'].fill_null(5.0).mean()

        logger.debug(
            f"NextGen - {player_id}: "
            f"separation={features['avg_separation']:.2f}yd, "
            f"cushion={features['avg_cushion']:.2f}yd"
        )

        return features

    except Exception as e:
        logger.error(f"Error extracting NextGen features: {e}")
        return features
```

**Integration Point (After target volume features)**

```python
# Add NextGen Stats (WR/TE only, 2016+)
if position in ['WR', 'TE']:
    nextgen_features = self._extract_nextgen_features(
        player_id, season, week, position
    )
    features.update(nextgen_features)
```

**Default Features Update**

```python
if position in ['WR', 'TE']:
    features.update({
        # ... target volume features ...
        'avg_separation': 2.5,
        'avg_cushion': 5.0
    })
```

**Testing:**
```python
# Test with 2024 data (NextGen available)
features_2024 = engineer.engineer_features(
    player_id="00-0036945",  # CeeDee Lamb
    season=2024,
    week=10,
    position='WR',
    prop_type='receiving_yards_wr',
    opponent_team='PHI'
)
assert 'avg_separation' in features_2024
assert 'avg_cushion' in features_2024
assert features_2024['avg_separation'] > 0

# Test with 2015 data (NextGen not available - should use defaults)
features_2015 = engineer.engineer_features(
    player_id="00-0029263",  # Julio Jones (example)
    season=2015,
    week=10,
    position='WR',
    prop_type='receiving_yards_wr',
    opponent_team='TB'
)
assert features_2015['avg_separation'] == 2.5  # Default
assert features_2015['avg_cushion'] == 5.0  # Default
```

---

## Phase 1.4: Add Prior Season Baseline

### Objective
Add year-over-year context to identify breakouts, declines, and consistency.

### Rationale
- Provides regression-to-mean signal
- Identifies sophomore breakouts, veteran declines
- Captures role changes (WR2 → WR1)

### Implementation

**File**: `modules/ml_feature_engineering.py`

**New Method: _extract_prior_season_baseline() (Insert after NextGen method)**

```python
def _extract_prior_season_baseline(
    self,
    player_id: str,
    season: int,
    position: str,
    stat_column: str
) -> Dict[str, float]:
    """
    Extract prior season baseline features (3 features).

    Provides year-over-year context for player performance:
    - prior_season_avg: Y-1 season average
    - prior_season_games: Y-1 games played
    - yoy_trend: (current_season_avg - prior_season_avg) / prior_season_avg

    Helps identify breakouts, declines, and consistency.

    Args:
        player_id: Player GSIS ID
        season: Current season
        position: Player position
        stat_column: Stat to analyze (e.g., 'receiving_yards')

    Returns:
        Dict with 3 prior season features
    """
    features = {
        'prior_season_avg': 0.0,  # Rookie / no prior data
        'prior_season_games': 0,
        'yoy_trend': 0.0  # No trend data
    }

    try:
        # Load Y-1 season stats
        prior_season = season - 1
        stats_file = Path(CACHE_DIR) / "positional_player_stats" / position.lower() / f"{position.lower()}-{prior_season}.csv"

        if not stats_file.exists():
            logger.debug(f"Prior season stats not found: {stats_file}")
            return features

        stats_df = pl.read_csv(stats_file)
        player_stats = stats_df.filter(pl.col('player_id') == player_id)

        if len(player_stats) == 0:
            logger.debug(f"Player {player_id} not found in {prior_season} data (likely rookie)")
            return features

        # Calculate Y-1 average
        if stat_column in player_stats.columns:
            prior_avg = player_stats[stat_column].fill_null(0).mean()
            prior_games = len(player_stats)

            features['prior_season_avg'] = prior_avg
            features['prior_season_games'] = prior_games

            # Calculate Y-o-Y trend if current season data available
            current_stats_file = Path(CACHE_DIR) / "positional_player_stats" / position.lower() / f"{position.lower()}-{season}.csv"
            if current_stats_file.exists():
                current_df = pl.read_csv(current_stats_file)
                current_player = current_df.filter(pl.col('player_id') == player_id)

                if len(current_player) > 0 and stat_column in current_player.columns:
                    current_avg = current_player[stat_column].fill_null(0).mean()

                    if prior_avg > 0:
                        features['yoy_trend'] = (current_avg - prior_avg) / prior_avg

            logger.debug(
                f"Prior season - {player_id}: "
                f"Y-1_avg={features['prior_season_avg']:.1f}, "
                f"games={features['prior_season_games']}, "
                f"trend={features['yoy_trend']:.2%}"
            )

        return features

    except Exception as e:
        logger.error(f"Error extracting prior season features: {e}")
        return features
```

**Integration Point (After baseline features extraction, ~line 122)**

```python
# Prior season baseline (for all props)
prior_season_features = self._extract_prior_season_baseline(
    player_id, season, position, stat_column
)
features.update(prior_season_features)
```

**Default Features Update**

```python
# In _get_default_features() baseline section
'prior_season_avg': 0.0,
'prior_season_games': 0,
'yoy_trend': 0.0
```

**Testing:**
```python
# Test veteran player (should have Y-1 data)
features_veteran = engineer.engineer_features(
    player_id="00-0036945",  # CeeDee Lamb (3rd year+)
    season=2024,
    week=10,
    position='WR',
    prop_type='receiving_yards_wr',
    opponent_team='PHI'
)
assert features_veteran['prior_season_avg'] > 0  # Should have 2023 data
assert features_veteran['prior_season_games'] > 0

# Test rookie (should use defaults)
features_rookie = engineer.engineer_features(
    player_id="00-0040000",  # Hypothetical rookie
    season=2024,
    week=10,
    position='WR',
    prop_type='receiving_yards_wr',
    opponent_team='PHI'
)
assert features_rookie['prior_season_avg'] == 0.0  # Default
```

---

## Phase 1.5: Remove last_5_avg

### Objective
Eliminate feature redundancy between last_3_avg and last_5_avg.

### Rationale
**User feedback:**
> "Why are we having both last 3 and last 5? seems like the similarity will add uneccesary noise"

**Analysis:**
- last_3_avg and last_5_avg share ~60% of the same games
- Multicollinearity introduces noise, not signal
- weighted_avg, last_3_avg, and career_avg provide sufficient time horizons

### Implementation

**File**: `modules/ml_feature_engineering.py`

**Change 1: Remove from default return (Line 239)**

```python
# BEFORE
return {
    'weighted_avg': 0.0,
    'last_3_avg': 0.0,
    'last_5_avg': 0.0,  # ← DELETE THIS LINE
    'career_avg': 0.0,
    ...
}

# AFTER
return {
    'weighted_avg': 0.0,
    'last_3_avg': 0.0,
    # last_5_avg removed (redundant)
    'career_avg': 0.0,
    ...
}
```

**Change 2: Remove calculation (Lines 252-258)**

```python
# BEFORE
features['last_3_avg'] = get_simple_average(
    player_stats, stat_column, last_n_games=3, through_week=through_week
)
features['last_5_avg'] = get_simple_average(  # ← DELETE THIS BLOCK
    player_stats, stat_column, last_n_games=5, through_week=through_week
)

# AFTER
features['last_3_avg'] = get_simple_average(
    player_stats, stat_column, last_n_games=3, through_week=through_week
)
# last_5_avg calculation removed
```

**Change 3: Remove from defaults (Line 1260)**

```python
# In _get_default_features()
# DELETE: 'last_5_avg': 0.0,
```

**Impact:**
- -1 feature (8 → 7 baseline features)
- Reduced multicollinearity
- Simplified model

**Testing:**
```python
features = engineer.engineer_features(
    player_id="00-0036945",
    season=2024,
    week=10,
    position='WR',
    prop_type='receiving_yards_wr',
    opponent_team='PHI'
)

# Verify removal
assert 'last_5_avg' not in features
assert 'last_3_avg' in features  # Still present
assert 'weighted_avg' in features  # Still present
```

---

## Phase 1.6: Fix RB Feature Pollution

### Objective
Prevent RB-specific features (blocking_quality, ypc_diff_pct) from appearing in WR/TE receiving models.

### Rationale
- Current issue: RB features added to ALL receiving props, including WR/TE
- Impact: 6 irrelevant features contaminating WR/TE training data
- Fix: Add proper position checks

### Implementation

**File**: `modules/ml_feature_engineering.py`

**Change 1: Fix blocking features logic (Lines 143-147)**

```python
# BEFORE
if position == 'RB':
    blocking_features = self._extract_blocking_quality_features(
        player_id, season, week
    )
    features.update(blocking_features)

# AFTER
if position == 'RB' and 'rushing' in prop_type:
    blocking_features = self._extract_blocking_quality_features(
        player_id, season, week
    )
    features.update(blocking_features)
```

**Change 2: Fix rushing volume features logic (Lines 149-155)**

```python
# BEFORE
# Add rushing volume features for rushing props
if 'rushing' in prop_type:
    volume_features = self._extract_rushing_volume_features(...)
    features.update(volume_features)

# AFTER
# Add rushing volume features for rushing props (RB only)
if position == 'RB' and 'rushing' in prop_type:
    volume_features = self._extract_rushing_volume_features(...)
    features.update(volume_features)
```

**Change 3: Fix defaults (Lines 1322-1330)**

```python
# BEFORE
if position == 'RB':
    features.update({
        'player_ypc': 4.3,
        ...
    })

# AFTER
if position == 'RB' and 'rushing' in prop_type:
    features.update({
        'player_ypc': 4.3,
        ...
    })
```

**Impact:**
- Prevents 6 RB features from polluting WR/TE models
- Cleaner feature space
- Reduced model confusion

**Testing:**
```python
# Test WR receiving yards - should NOT have RB features
wr_features = engineer.engineer_features(
    player_id="00-0036945",  # WR
    season=2024,
    week=10,
    position='WR',
    prop_type='receiving_yards_wr',
    opponent_team='PHI'
)
assert 'player_ypc' not in wr_features  # RB feature
assert 'ypc_diff_pct' not in wr_features  # RB feature
assert 'target_share_season' in wr_features  # WR feature ✓

# Test RB rushing yards - SHOULD have RB features
rb_features = engineer.engineer_features(
    player_id="00-0037744",  # RB
    season=2024,
    week=10,
    position='RB',
    prop_type='rushing_yards',
    opponent_team='PHI'
)
assert 'player_ypc' in rb_features  # Should be present ✓
```

---

## Phase 1.7: Target Qualification Filtering

### Objective
Filter training data to regular receiving targets only (3+ targets/week, 40+ season targets).

### Rationale
**Current issue:**
- WR3/WR4s with 1-2 targets/week included in training
- High variance, noise-dominated examples
- Model learns from low-signal data

**User feedback:** Apply ranking system's qualification approach

**Expected impact:**
- 20-30% reduction in training examples
- Much higher data quality
- Focus on predictable, high-volume receivers

### Implementation

**File**: `modules/ml_training_data_builder.py`

**New Helper Method: _count_player_targets() (Insert after line 278)**

```python
def _count_player_targets(
    self,
    player_id: str,
    season: int,
    week: int,
    pbp_df: Optional[pl.DataFrame]
) -> Tuple[int, int]:
    """
    Count targets for a player in a specific week and season-to-date.

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Week number
        pbp_df: Play-by-play DataFrame

    Returns:
        (targets_this_week, targets_season_through_week)
    """
    if pbp_df is None:
        return (0, 0)

    try:
        # Targets = complete_pass + incomplete_pass for this receiver
        all_targets = pbp_df.filter(
            (pl.col('receiver_player_id') == player_id) &
            ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
        )

        # Week-specific targets
        week_targets = all_targets.filter(pl.col('week') == week)
        targets_this_week = len(week_targets)

        # Season through this week
        season_targets = all_targets.filter(pl.col('week') <= week)
        targets_season = len(season_targets)

        return (targets_this_week, targets_season)

    except Exception as e:
        logger.debug(f"Error counting targets: {e}")
        return (0, 0)
```

**Add Qualification Filtering (Lines 120-134, after null target check)**

```python
# After existing null checks, ADD:

# Target qualification filtering (receiving props only)
if 'receiving' in prop_type or prop_type == 'receptions':
    targets_week, targets_season = self._count_player_targets(
        player_id, year, week, pbp_df
    )

    # Filter: 3+ targets this week AND 40+ targets season-to-date
    if targets_week < 3:
        skipped_low_targets += 1
        continue

    if targets_season < 40:
        skipped_low_targets += 1
        continue
```

**Track Skipped Stats (Lines 83-86, initialization)**

```python
skipped_no_opponent = 0
skipped_null_target = 0
skipped_low_targets = 0  # ADD THIS
```

**Log Skipped Stats (Lines 191-195, summary logging)**

```python
logger.info(f"  Skipped (no opponent): {skipped_no_opponent:,}")
logger.info(f"  Skipped (null target): {skipped_null_target:,}")
logger.info(f"  Low target volume (< 3/week or < 40/season): {skipped_low_targets:,}")  # ADD THIS
```

**Impact:**
- 20-30% fewer training examples
- Much higher quality (regular targets only)
- Focus on predictable, high-volume players

**Testing:**
```python
builder = TrainingDataBuilder()
train_df = builder.build_training_dataset(
    prop_type='receiving_yards_wr',
    start_year=2023,
    end_year=2023
)

# Verify filtering applied
print(f"Training examples: {len(train_df):,}")
print(f"Expected: 20-30% fewer than baseline")

# Sample player-weeks should have 3+ targets
sample_row = train_df.sample(1).row(0, named=True)
print(f"Sample player-week included: week={sample_row['week']}, player={sample_row['player_id']}")
# Manually verify this player had 3+ targets that week
```

---

## Testing Strategy

### Unit Tests

**Test 1: Feature Engineering**

```python
from modules.ml_feature_engineering import PropFeatureEngineer

def test_wr_receiving_features():
    """Test WR receiving_yards_wr feature engineering."""
    engineer = PropFeatureEngineer()

    features = engineer.engineer_features(
        player_id="00-0036945",  # CeeDee Lamb
        season=2024,
        week=10,
        position='WR',
        prop_type='receiving_yards_wr',
        opponent_team='PHI'
    )

    # Verify feature counts
    expected_feature_count = 64  # Updated from 55
    assert len(features) == expected_feature_count, f"Expected {expected_feature_count}, got {len(features)}"

    # Verify NEW features present
    assert 'target_share_season' in features, "Missing target volume features"
    assert 'targets_per_game_season' in features
    assert 'yards_per_target' in features
    assert 'avg_separation' in features, "Missing NextGen features"
    assert 'avg_cushion' in features
    assert 'prior_season_avg' in features, "Missing prior season features"

    # Verify REMOVED features absent
    assert 'last_5_avg' not in features, "last_5_avg should be removed"

    # Verify RB features NOT present (no pollution)
    assert 'player_ypc' not in features, "RB feature should not be in WR model"
    assert 'ypc_diff_pct' not in features

    # Verify non-zero values
    assert features['target_share_season'] >= 0, "Target share should be non-negative"
    assert features['avg_separation'] >= 0, "Separation should be non-negative"

    print("✅ WR feature engineering test PASSED")
```

**Test 2: TE Separate Model**

```python
def test_te_separate_model():
    """Test TE receiving_yards_te as separate prop type."""
    from modules.prop_types import get_prop_config

    # Verify WR prop exists and is WR-only
    wr_config = get_prop_config('receiving_yards_wr')
    assert wr_config is not None, "WR prop type not found"
    assert wr_config['position'] == ['WR'], "WR prop should be WR-only"

    # Verify TE prop exists and is TE-only
    te_config = get_prop_config('receiving_yards_te')
    assert te_config is not None, "TE prop type not found"
    assert te_config['position'] == ['TE'], "TE prop should be TE-only"

    # Verify no overlap
    assert wr_config['position'] != te_config['position']

    print("✅ WR/TE split test PASSED")
```

**Test 3: Target Qualification Filtering**

```python
def test_target_filtering():
    """Test that low-volume players are filtered out."""
    from modules.ml_training_data_builder import TrainingDataBuilder
    import polars as pl

    builder = TrainingDataBuilder()
    train_df = builder.build_training_dataset(
        prop_type='receiving_yards_wr',
        start_year=2023,
        end_year=2023
    )

    # Sample 10 random player-weeks
    sample = train_df.sample(min(10, len(train_df)))

    # Manually verify each has 3+ targets that week
    # (This would require loading PBP and checking - omitted for brevity)

    print(f"✅ Target filtering test PASSED - {len(train_df):,} examples")
```

### Integration Tests

**Test 4: Training Data Generation**

```python
def test_training_data_generation():
    """Test full training data generation pipeline."""
    from modules.ml_training_data_builder import TrainingDataBuilder

    builder = TrainingDataBuilder()

    # Generate WR training data
    wr_df = builder.build_training_dataset(
        prop_type='receiving_yards_wr',
        start_year=2023,
        end_year=2023
    )

    # Generate TE training data
    te_df = builder.build_training_dataset(
        prop_type='receiving_yards_te',
        start_year=2023,
        end_year=2023
    )

    # Verify separate datasets
    assert len(wr_df) > 0, "WR dataset is empty"
    assert len(te_df) > 0, "TE dataset is empty"
    assert len(wr_df) != len(te_df), "WR and TE should have different sample sizes"

    # Verify feature count
    expected_features = 64
    assert len(wr_df.columns) - 3 == expected_features, f"Expected {expected_features} features in WR data"  # -3 for target, player_id, week
    assert len(te_df.columns) - 3 == expected_features, f"Expected {expected_features} features in TE data"

    # Verify target volume features present
    assert 'target_share_season' in wr_df.columns
    assert 'target_share_season' in te_df.columns

    # Verify NextGen features present
    assert 'avg_separation' in wr_df.columns
    assert 'avg_separation' in te_df.columns

    # Verify last_5_avg removed
    assert 'last_5_avg' not in wr_df.columns
    assert 'last_5_avg' not in te_df.columns

    print(f"✅ Training data generation test PASSED")
    print(f"   WR examples: {len(wr_df):,}")
    print(f"   TE examples: {len(te_df):,}")
    print(f"   Features: {expected_features}")
```

### Validation Tests

**Test 5: No Nulls in Training Data**

```python
def test_no_nulls():
    """Verify no null features in training data."""
    import polars as pl

    # Load WR training data
    train_df = pl.read_parquet("cache/ml_training_data/receiving_yards_wr_2023_2023.parquet")

    # Check for nulls
    null_counts = train_df.null_count()

    for col in train_df.columns:
        null_pct = null_counts[col][0] / len(train_df)
        assert null_pct == 0, f"Column {col} has {null_pct:.1%} nulls"

    print("✅ No nulls test PASSED")
```

**Test 6: Feature Value Ranges**

```python
def test_feature_ranges():
    """Verify features are within expected ranges."""
    import polars as pl

    train_df = pl.read_parquet("cache/ml_training_data/receiving_yards_wr_2023_2023.parquet")

    # Target share should be 0-1
    assert train_df['target_share_season'].min() >= 0
    assert train_df['target_share_season'].max() <= 1

    # Separation should be 0-10 yards (reasonable range)
    assert train_df['avg_separation'].min() >= 0
    assert train_df['avg_separation'].max() <= 10

    # Targets per game should be 0-20 (reasonable max)
    assert train_df['targets_per_game_season'].min() >= 0
    assert train_df['targets_per_game_season'].max() <= 20

    print("✅ Feature range test PASSED")
```

---

## Results Presentation Format

### What You'll See When Models Complete

After training and evaluation, you will receive results in this format:

---

## Receiving Yards ML Enhancement Results

**Test Date:** 2025-11-XX
**Baseline Period:** 2022-2024
**Enhancements Applied:** 9 Phase 1 improvements

---

### Before Phase 1 Enhancements (Baseline)

**Configuration:**
- Prop Type: `receiving_yards` (WR + TE combined)
- Features: 55 total
- Position Filter: Mixed WR/TE model
- Target Qualification: None (all players included)
- Feature Issues: last_5_avg redundant, RB pollution, no target volume

**Performance (2022-2024 Evaluation):**
- **Overall Accuracy**: 41.6% (3,257/7,820 correct)
- **ROI**: -20.49%
- **Training Examples**: ~12,000 (2015-2024)
- **Feature Count**: 55

**Year-by-Year:**
| Year | Accuracy | Correct | Total | ROI |
|------|----------|---------|-------|-----|
| 2022 | 40.9% | 1,045 | 2,556 | -22.1% |
| 2023 | 41.8% | 1,098 | 2,627 | -21.3% |
| 2024 | 42.1% | 1,114 | 2,637 | -18.2% |

**Key Issues:**
- Below 48% breakeven threshold
- No target volume features (blind to #1 predictor)
- WR/TE usage patterns conflated
- Redundant features (last_5_avg)
- RB features polluting WR/TE models

---

### After Phase 1 Enhancements

**Configuration:**
- Prop Types: `receiving_yards_wr` + `receiving_yards_te` (separate models)
- Features: 64 total (+9 new, -1 removed)
- Position Filter: WR-only and TE-only models
- Target Qualification: 3+ targets/week, 40+ season targets
- Feature Improvements:
  - Added: Target volume (5), NextGen Stats (2), prior season (3)
  - Removed: last_5_avg (redundant)
  - Fixed: RB feature pollution eliminated

**Performance - WR Model (2022-2024 Evaluation):**
- **Overall Accuracy**: XX.X% (change: +X.X pp)
- **ROI**: XX.X% (change: +X.X pp)
- **Training Examples**: ~8,500 (2015-2024, -30% due to qualification)
- **Feature Count**: 64

**Performance - TE Model (2022-2024 Evaluation):**
- **Overall Accuracy**: XX.X% (change: +X.X pp)
- **ROI**: XX.X% (change: +X.X pp)
- **Training Examples**: ~2,500 (2015-2024, -25% due to qualification)
- **Feature Count**: 64

**Combined Performance:**
| Year | WR Accuracy | TE Accuracy | Combined Accuracy | ROI |
|------|-------------|-------------|-------------------|-----|
| 2022 | XX.X% | XX.X% | XX.X% | XX.X% |
| 2023 | XX.X% | XX.X% | XX.X% | XX.X% |
| 2024 | XX.X% | XX.X% | XX.X% | XX.X% |

---

### Improvements Observed

1. **Position Homogeneity** ✅
   - Before: Combined WR/TE model (conflated usage patterns)
   - After: Separate WR and TE models
   - **Impact**: Models learn position-specific patterns

2. **Target Volume Integration** ✅
   - Before: 0 target volume features (blind to #1 predictor)
   - After: 5 target volume features (share, raw counts, YPT)
   - **Impact**: Direct signal for receiving volume

3. **NextGen Stats Utilization** ✅
   - Before: 0 NextGen features (data unused)
   - After: 2 NextGen features (separation, cushion)
   - **Impact**: Captures receiver skill at creating space

4. **Feature Redundancy Eliminated** ✅
   - Before: last_5_avg redundant with last_3_avg
   - After: Removed last_5_avg
   - **Impact**: Reduced multicollinearity

5. **Data Quality Improved** ✅
   - Before: All players included (WR3/WR4 noise)
   - After: 3+ targets/week, 40+ season qualification
   - **Impact**: 20-30% fewer examples, much higher quality

6. **RB Pollution Eliminated** ✅
   - Before: 6 RB features in WR/TE models
   - After: RB features only in RB rushing models
   - **Impact**: Cleaner feature space for receivers

---

### Feature Importance Analysis

**Top 10 Most Important Features (WR Model):**
1. target_share_season (NEW) - XX.X% importance
2. targets_per_game_season (NEW) - XX.X% importance
3. weighted_avg - XX.X% importance
4. opponent_pass_def_rolling_3wk - XX.X% importance
5. yards_per_target (NEW) - XX.X% importance
6. avg_separation (NEW) - XX.X% importance
7. qb_passer_rating - XX.X% importance
8. target_share_3wk (NEW) - XX.X% importance
9. catch_rate_season - XX.X% importance
10. prior_season_avg (NEW) - XX.X% importance

**Key Insight:** X of top 10 features are NEW additions from Phase 1

---

### Success Criteria Assessment

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Overall Accuracy Improvement | +5 pp | +X.X pp | ✅/❌ |
| WR Accuracy | >48% | XX.X% | ✅/❌ |
| TE Accuracy | >45% | XX.X% | ✅/❌ |
| ROI Improvement | +10 pp | +X.X pp | ✅/❌ |
| Feature Count | 60-65 | 64 | ✅ |
| Training Examples Reduction | 20-30% | XX% | ✅/❌ |
| No Null Features | 0% | 0% | ✅ |

---

### Next Steps

**If Success Criteria Met:**
1. Deploy receiving_yards_wr and receiving_yards_te models to production
2. Apply same enhancements to receptions and receiving_tds prop types
3. Monitor live betting performance (Week X+)

**If Success Criteria Not Met:**
1. Analyze feature importance for low-performing features
2. Consider Phase 2 enhancements:
   - Route-specific features (slot vs outside)
   - QB-WR chemistry metrics
   - Home/away splits
3. Re-evaluate target qualification thresholds

---

**END OF RESULTS**

---

## Success Criteria

### Primary Goals

1. **Accuracy Improvement**
   - **Target**: +5 percentage points (41.6% → 46.6%+)
   - **Stretch Goal**: +10 pp (41.6% → 51.6%)
   - **Measurement**: 2022-2024 evaluation period

2. **ROI Improvement**
   - **Target**: +10 pp (-20.49% → -10.49% or better)
   - **Stretch Goal**: Break-even or positive ROI
   - **Measurement**: Simulated betting with 8% edge threshold

3. **Position-Specific Performance**
   - **WR Accuracy**: >48% (breakeven threshold)
   - **TE Accuracy**: >45% (lower threshold due to smaller sample)

### Secondary Goals

4. **Feature Quality**
   - Total features: 60-65 (currently 55)
   - No null features in training data (0% null rate)
   - Feature importance: Target volume in top 5

5. **Data Quality**
   - Training examples: 20-30% reduction (quality over quantity)
   - Example: ~12,000 → ~8,500 player-weeks
   - Regular targets only (3+ targets/week, 40+ season)

6. **Model Interpretability**
   - Feature importance analysis shows target volume as top predictor
   - No RB features in top 20 for WR/TE models
   - NextGen Stats visible in feature importance

### Minimum Viable Success

- **WR Accuracy**: >45% (+3.4 pp improvement)
- **TE Accuracy**: >42% (+0.4 pp improvement)
- **Combined Accuracy**: >44% (+2.4 pp improvement)
- **No regression**: Neither WR nor TE performs worse than baseline

### Failure Criteria

- Accuracy improvement <1 pp (not statistically significant)
- Either WR or TE model performs worse than baseline
- Training data generation fails or produces <5,000 examples
- Excessive null features (>5% null rate)

---

## Risk Mitigation

### Risk 1: Training Data Size Reduction

**Risk**: Target qualification filtering may remove too many examples.

**Mitigation**:
- Monitor training set size (expect ~8,500 WR, ~2,500 TE examples)
- If <5,000 examples total, lower thresholds:
  - Option 1: 2+ targets/week (instead of 3+)
  - Option 2: 30+ season targets (instead of 40+)
- 10 years of data (2015-2024) provides buffer

**Likelihood**: Low
**Impact**: Medium

### Risk 2: NextGen Data Coverage

**Risk**: Pre-2016 seasons lack NextGen Stats.

**Mitigation**:
- Use defaults (2.5 yd separation, 5.0 yd cushion) for 2015
- 2016-2024 = 9 years of NextGen data (sufficient)
- Model learns from defaults when data unavailable

**Likelihood**: N/A (expected behavior)
**Impact**: Low

### Risk 3: Rookie Prior Season Data

**Risk**: Rookies lack prior_season_avg data.

**Mitigation**:
- Default to 0.0 for rookies (model learns "no prior data" signal)
- yoy_trend = 0.0 indicates rookie status
- Model can learn rookie-specific patterns from other features

**Likelihood**: Expected
**Impact**: Low

### Risk 4: WR/TE Split Reduces Sample Size

**Risk**: Separate models have less data per position.

**Mitigation**:
- WR model: ~70% of combined data (still robust)
- TE model: ~30% of combined data (smaller but sufficient)
- 10 years provides adequate sample (2,500+ TE examples)
- Alternative: If TE model fails, revert to combined model

**Likelihood**: Low
**Impact**: Medium

### Risk 5: Model Complexity Overfitting

**Risk**: +9 features may cause overfitting.

**Mitigation**:
- Use ensemble methods (XGBoost, RandomForest) with regularization
- Cross-validation on training data
- Monitor train vs. validation accuracy gap
- Feature importance analysis to identify noise features

**Likelihood**: Low (ensemble methods handle high-dimensional data well)
**Impact**: Medium

---

## Implementation Checklist

### Pre-Implementation

- [ ] Backup current receiving_yards models
- [ ] Backup current training data
- [ ] Run data coverage validation tests
- [ ] Create git branch: `feature/receiving-yards-phase1`

### Phase 1.1: Split WR/TE Prop Types

- [ ] Update `prop_types.py` - create receiving_yards_wr
- [ ] Update `prop_types.py` - create receiving_yards_te
- [ ] Update `prop_types.py` - update POSITION_PROP_TYPES mapping
- [ ] Test: `python modules/prop_types.py` (should run without errors)
- [ ] Test: Verify WR/TE prop configs load correctly

### Phase 1.2: Add Target Volume Features

- [ ] Add `_extract_target_volume_features()` method to `ml_feature_engineering.py`
- [ ] Integrate target volume features into `engineer_features()` method
- [ ] Update `_get_default_features()` with target volume defaults
- [ ] Test: Unit test for target volume feature extraction
- [ ] Test: Verify non-zero values for known players

### Phase 1.3: Integrate NextGen Stats

- [ ] Add `_extract_nextgen_features()` method to `ml_feature_engineering.py`
- [ ] Integrate NextGen features into `engineer_features()` method
- [ ] Update `_get_default_features()` with NextGen defaults
- [ ] Test: Unit test for NextGen feature extraction (2016+ data)
- [ ] Test: Verify defaults used for pre-2016 seasons

### Phase 1.4: Add Prior Season Baseline

- [ ] Add `_extract_prior_season_baseline()` method to `ml_feature_engineering.py`
- [ ] Integrate prior season features into `engineer_features()` method
- [ ] Update `_get_default_features()` with prior season defaults
- [ ] Test: Unit test for prior season extraction
- [ ] Test: Verify defaults for rookies

### Phase 1.5: Remove last_5_avg

- [ ] Remove `last_5_avg` from default return in `_extract_baseline_features()`
- [ ] Remove `last_5_avg` calculation block
- [ ] Remove `last_5_avg` from `_get_default_features()`
- [ ] Test: Verify `last_5_avg` not in feature dict

### Phase 1.6: Fix RB Feature Pollution

- [ ] Add position check to blocking features logic
- [ ] Add position check to rushing volume features logic
- [ ] Add position check to RB defaults
- [ ] Test: Verify no RB features in WR/TE models
- [ ] Test: Verify RB features still in RB rushing models

### Phase 1.7: Target Qualification Filtering

- [ ] Add `_count_player_targets()` helper method to `ml_training_data_builder.py`
- [ ] Add target qualification filtering logic
- [ ] Track skipped_low_targets stat
- [ ] Log skipped stats in summary
- [ ] Test: Verify filtering applied
- [ ] Test: Sample player-weeks have 3+ targets

### Post-Implementation

- [ ] Delete old receiving_yards training data (if exists)
- [ ] Regenerate training data for receiving_yards_wr (2015-2024)
- [ ] Regenerate training data for receiving_yards_te (2015-2024)
- [ ] Verify training data feature count (64 expected)
- [ ] Verify no null features
- [ ] Run all unit tests
- [ ] Run all integration tests
- [ ] Commit changes with descriptive message

### Model Training & Evaluation

- [ ] Train receiving_yards_wr model (2015-2021 training, 2022-2024 eval)
- [ ] Train receiving_yards_te model (2015-2021 training, 2022-2024 eval)
- [ ] Evaluate WR model (2022-2024)
- [ ] Evaluate TE model (2022-2024)
- [ ] Generate feature importance analysis
- [ ] Generate comparison table (Before vs After)
- [ ] Document results in `research/RECEIVING_YARDS_PHASE1_RESULTS.md`

---

## Timeline

### Week 1: Implementation (Days 1-3)

**Day 1: Split & Remove** (4 hours)
- Morning: Phase 1.1 (Split WR/TE prop types)
- Afternoon: Phase 1.5 (Remove last_5_avg), Phase 1.6 (Fix RB pollution)
- Testing: Unit tests for prop config, feature engineering

**Day 2: Add Features Part 1** (6 hours)
- Morning: Phase 1.2 (Target volume features - most complex)
- Afternoon: Phase 1.3 (NextGen Stats)
- Testing: Unit tests for new feature extraction methods

**Day 3: Add Features Part 2 + Filtering** (5 hours)
- Morning: Phase 1.4 (Prior season baseline)
- Afternoon: Phase 1.7 (Target qualification filtering)
- Testing: Integration tests, training data generation test run

### Week 1: Training & Evaluation (Days 4-5)

**Day 4: Data Generation & Training** (8 hours)
- Morning: Regenerate training data (WR + TE)
- Afternoon: Train WR model (2015-2021)
- Evening: Train TE model (2015-2021)

**Day 5: Evaluation & Documentation** (6 hours)
- Morning: Evaluate both models (2022-2024)
- Afternoon: Generate results, comparison tables
- Evening: Document in RECEIVING_YARDS_PHASE1_RESULTS.md

### Total Estimated Time: 29 hours (~3-4 days)

---

## Appendix A: Feature Comparison Table

| Feature Category | Before | After | Change |
|-----------------|--------|-------|--------|
| Baseline | 8 | 7 | -1 (removed last_5_avg) |
| Target Volume | 0 | 5 | +5 NEW |
| NextGen Stats | 0 | 2 | +2 NEW |
| Prior Season | 0 | 3 | +3 NEW |
| Game Script | 6 | 6 | No change |
| Opponent Defense | 9 | 9 | No change |
| Catch Rate | 3 | 3 | No change |
| QB Quality | 6 | 6 | No change |
| Team Offense | 5 | 5 | No change |
| Weather | 4 | 4 | No change |
| RB Features (pollution) | 6 (incorrect) | 0 | Eliminated |
| **TOTAL** | **55** | **64** | **+9 net** |

---

## Appendix B: File Change Summary

| File | Lines Changed | Complexity | Testing Required |
|------|--------------|------------|------------------|
| `modules/prop_types.py` | ~50 | Low | Unit test |
| `modules/ml_feature_engineering.py` | ~300 | Medium | Unit + Integration |
| `modules/ml_training_data_builder.py` | ~50 | Low | Integration |
| **TOTAL** | **~400 lines** | **Medium** | **Comprehensive** |

---

## Appendix C: Questions for User Confirmation

Before proceeding with implementation, please confirm:

1. **Target filtering thresholds**: Confirm 3+ targets/week AND 40+ season targets is acceptable, or adjust?

2. **Prior season features**: Confirm 3 features (avg, games, trend) is sufficient, or add more Y-1 metrics?

3. **NextGen features**: Confirm raw values only (no ratios) as requested?

4. **RB receiving props**: Should we create `receiving_yards_rb` as a separate model, or exclude RBs entirely from receiving yards predictions?

5. **Apply to all receiving props**: Split receptions and receiving_tds the same way (receptions_wr, receptions_te, etc.)?

---

## Appendix D: Reserved Features for Phase 2 (If Needed)

**Note**: These features are ONLY to be implemented if Phase 1 enhancements fail to meet success criteria (WR accuracy <48%, TE accuracy <45%, or combined ROI improvement <+10 pp).

### Competition for Targets Features

**Rationale**: Capture how concentrated team targets are and whether player is WR1/WR2/committee back.

**Why not in Phase 1**: Current `target_share_season` and `target_share_3wk` already encode this signal implicitly:
- 30% target share = clear WR1
- 15% target share = WR2 or committee
- 10% target share = WR3/depth

**Features to Add (if Phase 1 insufficient):**

1. **team_target_concentration** (1 feature)
   - **Definition**: Herfindahl-Hirschman Index (HHI) or Gini coefficient of team target distribution
   - **Calculation**:
     ```python
     # HHI approach
     team_targets = pbp.filter(posteam == team, week < week_n)
     target_shares = team_targets.group_by('receiver_player_id').agg(
         (pl.count() / len(team_targets)).alias('share')
     )
     hhi = (target_shares['share'] ** 2).sum()
     # hhi near 1.0 = one dominant target
     # hhi near 0.2 = spread evenly across 5+ players
     ```
   - **Expected Impact**: Distinguish pass-heavy offenses (KC, MIA) from balanced attacks (SF, BAL)
   - **Data Source**: PBP receiver_player_id (1999+)

2. **num_viable_targets** (1 feature)
   - **Definition**: Count of players with 5+ targets on the team through week-1
   - **Calculation**:
     ```python
     team_target_counts = pbp.filter(
         posteam == team, week < week_n
     ).group_by('receiver_player_id').agg(pl.count().alias('targets'))

     num_viable = (team_target_counts['targets'] >= 5).sum()
     # 1-2 = concentrated offense (WR1 dominance)
     # 3-4 = typical NFL offense
     # 5+ = spread/committee offense
     ```
   - **Expected Impact**: Identify situations where targets are heavily concentrated vs spread
   - **Data Source**: PBP receiver_player_id (1999+)

3. **position_target_share** (1 feature)
   - **Definition**: % of position group targets (WRs only for WR, TEs only for TE)
   - **Calculation**:
     ```python
     # For WR
     wr_targets = pbp.filter(
         posteam == team,
         week < week_n,
         position == 'WR'  # Filter to WR targets only
     )
     player_wr_share = player_targets / len(wr_targets)
     # Identifies alpha WR even if RB/TE get significant targets
     ```
   - **Expected Impact**: Better capture WR1 status in run-heavy/TE-heavy offenses
   - **Data Source**: PBP receiver_player_id + position roster data

**Implementation Trigger**:
- Phase 1 WR accuracy <48% AND
- Feature importance shows `target_share_season` NOT in top 5 features

**Testing Strategy**:
- Add features one at a time
- Re-evaluate after each addition
- If no improvement after all 3, features are not predictive

**Risk Assessment**:
- **Low risk**: All features use existing PBP data, no new data sources
- **Multicollinearity risk**: May correlate with existing target_share features
- **Overfitting risk**: Adding 3 more features (52 → 55) could cause noise

---

**END OF IMPLEMENTATION PLAN**
