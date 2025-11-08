# WR Receiving Yards ML Model - Feature Cleanup & Enhancement Plan

**Date**: 2025-01-06
**Status**: Implementation Plan
**Current Performance**: 41.6% accuracy, -20.49% ROI (below 48% breakeven)
**Goal**: Clean up broken/redundant features, add missing high-value features, reach breakeven threshold

---

## Executive Summary

The WR receiving yards ML model currently uses **49 features** but suffers from multiple critical issues:
- **1 broken feature** that always returns 0.0 (catch_rate_over_exp)
- **7 features using hardcoded defaults** instead of real data (game context)
- **12 injury features** vs 1 in profitable RB model (noise/overfitting)
- **2 redundant features** (last_3_avg, success_rate_season)
- **NextGen stats using synthetic defaults** for 2009-2015 (adds noise)

This plan reduces to **~35 clean features** with proper implementations and NaN handling for partial-year data.

---

## Current Feature Inventory (49 features)

### Baseline Performance (7 features)
1. `weighted_avg` - Recency-weighted rolling average (L3: 1.5x, L4-6: 1.0x, L7+: 0.75x)
2. `last_3_avg` - Simple 3-game average **[REDUNDANT - REMOVE]**
3. `career_avg` - 3-year historical lookback average
4. `variance_cv` - Coefficient of variation (consistency)
5. `games_played` - Sample size indicator
6. `effective_games` - Quality-adjusted games (QB only)
7. `effective_games_ratio` - Ratio metric (QB only)

### Opponent Defense (2 features)
8. `opp_def_pass_ypa` - Opponent's yards per pass attempt allowed
9. `opp_def_pass_td_rate` - Opponent's TD rate allowed

### Efficiency Metrics (4 features)
10. `success_rate_3wk` - 3-week rolling success rate
11. `success_rate_season` - Season-long success rate **[REDUNDANT - REMOVE]**
12. `red_zone_rate` - Percentage of touches inside opponent 20-yard line
13. `usage_rate` - Target/carry share

### Catch Rate (4 features)
14. `catch_rate` - Completions / targets
15. `catch_rate_over_exp` - Catch rate vs expected **[BROKEN - ALWAYS 0.0 - REMOVE]**
16. `avg_target_depth` - Average air yards per target
17. `yac_pct` - Yards after catch as % of total yards

### Target Volume (6 features)
18. `targets_season_avg` - Season average targets
19. `targets_3wk_avg` - 3-week rolling targets
20. `target_share_season` - % of team targets (season)
21. `target_share_3wk` - % of team targets (3-week)
22. `yards_per_target_season` - Season Y/T efficiency
23. `yards_per_target_3wk` - 3-week Y/T efficiency

### NextGen Stats (2 features)
24. `avg_separation` - GPS-tracked separation at catch **[USING DEFAULTS PRE-2016]**
25. `avg_cushion` - Yards of cushion at snap **[USING DEFAULTS PRE-2016]**

### Injury Features (12 features)
26. `injury_games_missed_y1` - Games missed last year **[NOISE - REMOVE]**
27. `injury_games_missed_y2` - Games missed 2 years ago **[NOISE - REMOVE]**
28. `injury_games_missed_y3` - Games missed 3 years ago **[NOISE - REMOVE]**
29. `injury_classification_score` - Injury-prone rating **[NOISE - REMOVE]**
30. `has_recurring_injury` - Recurring injury flag **[NOISE - REMOVE]**
31. `games_missed_current_season` - Games missed this season **[NOISE - REMOVE]**
32. `injury_status_score` - Current week status (0=healthy, 3=out) **[KEEP - ONLY ONE NEEDED]**
33. `injury_type_mobility` - Lower body injury flag **[NOISE - REMOVE]**
34. `injury_type_upper_body` - Upper body injury flag **[NOISE - REMOVE]**
35. `weeks_since_last_missed` - Recency metric **[NOISE - REMOVE]**
36. `is_on_injury_report` - Boolean flag **[NOISE - REMOVE]**
37. (Additional injury metadata) **[NOISE - REMOVE]**

### Prior Season (3 features)
38. `prior_season_avg` - Prior year average
39. `yoy_trend` - Year-over-year trend
40. `sophomore_indicator` - Second year flag

### Game Context (7 features)
41. `is_home` - Home game indicator **[USING DEFAULT 0.5 - IMPLEMENT]**
42. `is_dome` - Dome/indoor indicator **[USING DEFAULT 0.0 - IMPLEMENT]**
43. `division_game` - Divisional opponent **[USING DEFAULT 0.0 - IMPLEMENT]**
44. `game_temp` - Temperature in Fahrenheit **[USING DEFAULT 70.0 - IMPLEMENT]**
45. `game_wind` - Wind speed in mph **[USING DEFAULT 5.0 - IMPLEMENT]**
46. `vegas_total` - Game over/under line **[USING DEFAULT 45.0 - IMPLEMENT]**
47. `vegas_spread` - Point spread **[USING DEFAULT 0.0 - IMPLEMENT]**

### Game Script (5 features)
48. `team_avg_margin` - Rolling 3-game point differential
49. `opp_def_ppg_allowed` - Opponent's PPG allowed
50. `team_plays_per_game` - Offensive plays per game (pace)
51. `team_time_of_possession` - Average TOP in minutes
52. `team_rb_quality` - Team RB YPC vs league avg **[IRRELEVANT FOR WR - REMOVE]**

### Categorical (4 features)
53. `opponent` - Opponent team code
54. `position` - Player position (WR/TE)
55. `week` - Week number
56. `season` - Season year

---

## Critical Issues Identified

### Issue 1: Broken Features
**File**: `modules/ml_feature_engineering.py:577`

**Feature**: `catch_rate_over_exp`
- **Problem**: Always returns 0.0 (never implemented)
- **Impact**: Adds pure noise to model
- **Solution**: Remove completely

### Issue 2: Hardcoded Defaults (Synthetic Data)
**File**: `modules/ml_feature_engineering.py:1455-1464`

**Features**: 7 game context features use hardcoded defaults for ALL predictions
```python
'is_home': 0.5,          # Should be 0 or 1 based on schedule
'is_dome': 0.0,          # Should be based on stadium type
'division_game': 0.0,    # Should be based on division matchups
'game_temp': 70.0,       # Should load from cache/weather_data/games_weather.csv
'game_wind': 5.0,        # Should load from cache/weather_data/games_weather.csv
'vegas_total': 45.0,     # Should load from betting lines cache
'vegas_spread': 0.0      # Should load from betting lines cache
```

**Impact**: Every prediction uses same values (no signal), adds pure noise

**Solution**: Implement proper data loading OR use NaN for missing data

### Issue 3: NextGen Stats Using Synthetic Defaults Pre-2016
**File**: `modules/ml_feature_engineering.py:971-978`

**Features**: `avg_separation`, `avg_cushion`
- **Problem**: Returns league averages (2.5, 5.0) for 2009-2015 instead of NaN
- **Impact**: Pollutes 2009-2015 training data with synthetic values
- **Solution**: Return `float('nan')` for pre-2016, let tree models handle intelligently

### Issue 4: Injury Feature Bloat
**Comparison**:
- **RB rushing (profitable)**: 1 injury feature (`injury_status_score`)
- **WR receiving (unprofitable)**: 12 injury features

**Problem**: Historical injury data (Y-1, Y-2, Y-3) adds noise without predictive value

**Solution**: Match RB model - use only `injury_status_score` (current week status)

### Issue 5: Redundant Features
**Redundant Pairs**:
1. `weighted_avg` vs `last_3_avg` - Both measure recent performance, high correlation
2. `success_rate_3wk` vs `success_rate_season` - Overlapping signal during late season

**Solution**: Remove `last_3_avg` and `success_rate_season`

---

## Feature Comparison: Profitable vs Unprofitable Models

| Category | RB Rushing (Profitable) | WR Receiving (Current) | WR Receiving (Planned) |
|----------|------------------------|------------------------|------------------------|
| Baseline | 5 | 7 | 4 |
| Opponent | 2 | 2 | 2 |
| Efficiency | 4 | 4 | 3 |
| Volume | 0 | 6 | 2 |
| Catch Rate | 0 | 4 | 3 |
| Injury | **1** | **12** | **1** |
| NextGen | 0 | 2 | 2 |
| Prior Season | 0 | 3 | 3 |
| Game Context | 7 | 7 | 7 |
| Game Script | 5 | 5 | 4 |
| Categorical | 4 | 4 | 4 |
| **TOTAL** | **31** | **49** | **~35** |

**Key Insight**: Profitable RB model uses 31 features. WR model has 58% MORE features (49), many adding noise.

---

## NextGen Stats NaN Handling Strategy

### The Problem
- NextGen data only available 2016-2024
- Current implementation returns league averages (2.5, 5.0) for 2009-2015
- Synthetic defaults pollute training data with fake signals

### The Solution: NaN Handling (Option C)

**Why This Works**:
1. **3 of 4 ensemble models support NaN natively**: XGBoost, LightGBM, CatBoost
2. **Tree models learn optimal splits**: "If NextGen available, use it; else use other features"
3. **No synthetic noise**: 2009-2015 gets NaN (not fake values)
4. **Preserves full dataset**: All years 2009-2024 included

**Implementation**:
```python
# modules/ml_feature_engineering.py, line 971-974
# BEFORE (Current - Synthetic Defaults):
features = {
    'avg_separation': 2.5,  # League average
    'avg_cushion': 5.0,     # League average
}

# AFTER (NaN for Missing):
features = {
    'avg_separation': float('nan'),
    'avg_cushion': float('nan'),
}
```

**How Models Handle NaN**:
- **XGBoost/LightGBM/CatBoost**: Treat NaN as separate category, learn optimal direction
- **Random Forest (sklearn)**: Requires imputation OR removal from ensemble
- **Preprocessing**: Separate NaN-friendly features from scaled features

**Expected Impact**:
- 2009-2015: NaN for NextGen (model uses other features)
- 2016-2024: Real NextGen values (model learns separation/cushion signal)
- Result: Better signal-to-noise ratio

---

## Implementation Plan

### Phase 1: Remove Broken/Unimplemented Features

**File**: `modules/ml_feature_engineering.py`

**Actions**:
1. Remove `catch_rate_over_exp` from `_extract_catch_rate_features()` (line 577)
2. Remove `last_3_avg` from baseline feature generation
3. Remove `success_rate_season` from efficiency feature generation
4. Remove `team_rb_quality` from game script features (irrelevant for WR)

**Expected Impact**: -4 noise features

---

### Phase 2: Fix NextGen Stats with NaN Handling

**File**: `modules/ml_feature_engineering.py`, lines 971-974

**Change**:
```python
# OLD
features = {
    'avg_separation': 2.5,
    'avg_cushion': 5.0,
}

# NEW
features = {
    'avg_separation': float('nan'),
    'avg_cushion': float('nan'),
}
```

**Test**: Verify 2015 samples have NaN, 2016+ have real values

---

### Phase 3: Implement Weather & Game Context Properly

**File**: `modules/ml_feature_engineering.py`, `_extract_game_context_features()`

**Current Problems**:
- All 7 features use hardcoded defaults
- Weather cache exists but not used
- Schedule/stadium data available but not loaded

**Required Implementations**:

#### 3.1 Weather Data Loading
```python
def _load_game_weather(self, season: int, week: int, team: str) -> Dict[str, float]:
    """Load actual game weather from cache."""
    weather_file = f"cache/weather_data/games_weather.csv"

    # Load and filter to game
    weather_df = pd.read_csv(weather_file)
    game_weather = weather_df[
        (weather_df['season'] == season) &
        (weather_df['week'] == week) &
        (weather_df['team'] == team)
    ]

    if game_weather.empty:
        return {'game_temp': float('nan'), 'game_wind': float('nan')}

    return {
        'game_temp': game_weather['temperature'].values[0],
        'game_wind': game_weather['wind_speed'].values[0]
    }
```

#### 3.2 Home/Away Detection
```python
def _get_home_away(self, season: int, week: int, team: str) -> float:
    """Determine if team is home or away."""
    # Load from PBP data or schedule cache
    # Filter to season, week, team
    # Return 1.0 for home, 0.0 for away, NaN if unknown
```

#### 3.3 Stadium Type
```python
def _get_stadium_type(self, season: int, team: str) -> float:
    """Get stadium type (dome/outdoor)."""
    # Load stadium info
    # Return 1.0 for dome, 0.0 for outdoor, NaN if unknown
```

#### 3.4 Division Matchup
```python
def _is_division_game(self, team: str, opponent: str, season: int) -> float:
    """Check if divisional matchup."""
    # Load division assignments
    # Return 1.0 if same division, 0.0 if not, NaN if unknown
```

#### 3.5 Vegas Lines
```python
def _load_vegas_lines(self, season: int, week: int, team: str) -> Dict[str, float]:
    """Load betting lines if available."""
    # Load from betting lines cache if exists
    # Return actual totals/spreads or NaN
```

**Critical Rule**: Use `float('nan')` for missing data, NOT hardcoded defaults

---

### Phase 4: Reduce Injury Features (Match RB Model)

**File**: `modules/ml_feature_engineering.py`, lines 1337-1343

**Current**: RB rushing uses 1 injury feature successfully

**Extend to WR/TE**:
```python
# Reduce noise: only current week status
if position == 'RB' and 'rushing' in prop_type:
    return {'injury_status_score': features.get('injury_status_score', 0.0)}

# ADD THIS:
if position in ['WR', 'TE'] and 'receiving' in prop_type:
    return {'injury_status_score': features.get('injury_status_score', 0.0)}

# All other positions/props get full suite (if needed)
return features
```

**Expected Impact**: -11 noise features (from 12 to 1)

---

### Phase 5: Verify/Add Target Volume Features

**File**: `modules/ml_feature_engineering.py`, `_extract_target_volume_features()`

**Required Features** (may be partially implemented):
1. `target_share_season` - % of team targets (season-long)
2. `targets_per_game` - Raw target count (3-week rolling)

**Data Source**: PBP data, filter to `receiver_player_id` where `pass_attempt == 1`

**Calculation**:
```python
# Target share
player_targets = pbp_data[pbp_data['receiver_player_id'] == player_id].shape[0]
team_targets = pbp_data[pbp_data['posteam'] == team].shape[0]
target_share_season = player_targets / team_targets if team_targets > 0 else 0.0

# Targets per game (3-week rolling)
recent_games = last_3_games[last_3_games['receiver_player_id'] == player_id]
targets_per_game = recent_games.groupby('game_id').size().mean()
```

**Note**: These may correlate but both provide value (relative vs absolute usage)

---

### Phase 6: Create Feature Filtering System

**File**: `modules/prop_types.py` (new section)

**Add Configuration**:
```python
PROP_FEATURE_CONFIG = {
    'receiving_yards': {
        'include': [
            # Baseline (4)
            'weighted_avg', 'career_avg', 'variance_cv', 'games_played',

            # Opponent (2)
            'opp_def_pass_ypa', 'opp_def_pass_td_rate',

            # Efficiency (3)
            'success_rate_3wk', 'red_zone_rate', 'usage_rate',

            # Injury (1)
            'injury_status_score',

            # Catch Rate (3)
            'catch_rate', 'avg_target_depth', 'yac_pct',

            # Target Volume (2)
            'target_share_season', 'targets_per_game',

            # NextGen (2) - NaN for pre-2016
            'avg_separation', 'avg_cushion',

            # Game Context (5)
            'is_home', 'is_dome', 'division_game', 'game_temp', 'game_wind',

            # Game Script (4)
            'team_avg_margin', 'opp_def_ppg_allowed', 'team_plays_per_game', 'team_time_of_possession',

            # Vegas (2)
            'vegas_total', 'vegas_spread',

            # Prior Season (3)
            'prior_season_avg', 'yoy_trend', 'sophomore_indicator',

            # Categorical (4)
            'opponent', 'position', 'week', 'season'
        ]
    },
    'rushing_yards': {
        # Keep existing successful feature set (31 features)
        # Don't modify - already profitable
    },
    'passing_yards': {
        # Keep existing feature set
    }
}
```

**File**: `modules/ml_ensemble.py` (after line 101)

**Add Filtering**:
```python
# Filter features based on prop type
from modules.prop_types import PROP_FEATURE_CONFIG

if prop_type in PROP_FEATURE_CONFIG:
    allowed_features = PROP_FEATURE_CONFIG[prop_type]['include']
    allowed_features.extend(['target', 'player_id', 'year'])  # Keep metadata

    # Filter training and test data
    X_train = X_train[[col for col in X_train.columns if col in allowed_features]]
    X_test = X_test[[col for col in X_test.columns if col in allowed_features]]
```

---

### Phase 7: Handle NaN in Preprocessing

**File**: `modules/ml_ensemble.py`, lines 96-101

**Problem**: StandardScaler fails with NaN values

**Solution: Selective Scaling**:
```python
# Features that may have NaN (let tree models handle)
nan_features = [
    'avg_separation', 'avg_cushion',  # NextGen (pre-2016)
    'game_temp', 'game_wind',  # Weather (if missing)
    'is_home', 'is_dome', 'division_game',  # Context (if missing)
    'vegas_total', 'vegas_spread'  # Betting lines (if missing)
]

# Get numeric columns
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Features to scale (exclude NaN-friendly features and metadata)
scale_features = [
    col for col in numeric_features
    if col not in nan_features and col not in ['target', 'player_id', 'year']
]

# Scale only non-NaN features
scaler = StandardScaler()
X_train[scale_features] = scaler.fit_transform(X_train[scale_features])
X_test[scale_features] = scaler.transform(X_test[scale_features])

# Leave NaN features as-is for tree models to handle naturally
# XGBoost, LightGBM, CatBoost all support native NaN handling
```

**Alternative**: Remove Random Forest from ensemble (sklearn doesn't handle NaN)

---

### Phase 8: Rebuild Training Data & Test

**8.1 Rebuild Training Data**:
```bash
source .venv/Scripts/activate
python modules/ml_training_data_builder.py --prop_type receiving_yards --years 2009-2024
```

**8.2 Verify Feature Quality**:
- Check 2009-2015 samples have NaN for NextGen features
- Check 2016+ samples have real NextGen values
- Verify no hardcoded defaults remain
- Confirm feature count is ~35 (down from 49)

**8.3 Retrain Model**:
```bash
python modules/ml_ensemble.py --prop_type receiving_yards
```

**8.4 Evaluate Performance**:
```bash
python tests/player_props/evaluate_prop_type.py --prop_type receiving_yards --year 2024
```

**8.5 Compare Results**:
- Baseline: 41.6% accuracy, -20.49% ROI
- Target: 48%+ accuracy (breakeven threshold)
- Goal: Positive ROI like RB rushing model

---

## Expected Results

### Before Cleanup (Current State)
**Feature Count**: 49 features

**Problems**:
- 1 broken feature (catch_rate_over_exp = 0.0)
- 7 features with hardcoded defaults (pure noise)
- 12 injury features (vs 1 in profitable RB model)
- 2 redundant features
- NextGen using synthetic defaults for 2009-2015
- No feature filtering system

**Performance**: 41.6% accuracy, -20.49% ROI

### After Cleanup (Planned State)
**Feature Count**: ~35 features

**Improvements**:
- 0 broken features
- 0 hardcoded defaults (all real data or NaN)
- 1 injury feature (match RB model)
- 0 redundant features
- NextGen using NaN for 2009-2015 (tree models handle intelligently)
- Feature filtering system for maintainability
- 2 target volume features added

**Expected Performance**: 48%+ accuracy (breakeven), improved ROI

---

## Implementation Checklist

- [ ] **Phase 1**: Remove broken/unimplemented features
  - [ ] Remove catch_rate_over_exp
  - [ ] Remove last_3_avg
  - [ ] Remove success_rate_season
  - [ ] Remove team_rb_quality

- [ ] **Phase 2**: Fix NextGen stats
  - [ ] Change defaults to float('nan')
  - [ ] Test on 2015 vs 2016 samples

- [ ] **Phase 3**: Implement game context properly
  - [ ] Weather data loading
  - [ ] Home/away detection
  - [ ] Stadium type
  - [ ] Division matchups
  - [ ] Vegas lines (if available)

- [ ] **Phase 4**: Reduce injury features
  - [ ] Extend RB filtering to WR/TE
  - [ ] Keep only injury_status_score

- [ ] **Phase 5**: Verify target volume features
  - [ ] Check if target_share_season exists
  - [ ] Check if targets_per_game exists
  - [ ] Implement missing features

- [ ] **Phase 6**: Create feature filtering system
  - [ ] Add PROP_FEATURE_CONFIG to prop_types.py
  - [ ] Add filtering logic to ml_ensemble.py
  - [ ] Test filtering works correctly

- [ ] **Phase 7**: Handle NaN in preprocessing
  - [ ] Implement selective scaling
  - [ ] Test with NaN values
  - [ ] Verify models train successfully

- [ ] **Phase 8**: Rebuild & test
  - [ ] Rebuild training data (2009-2024)
  - [ ] Verify feature quality
  - [ ] Retrain model
  - [ ] Evaluate on 2024 season
  - [ ] Compare vs baseline performance

---

## Success Criteria

1. **Feature count reduced**: From 49 to ~35 features
2. **No broken features**: All features return real data or NaN (not hardcoded defaults)
3. **Model trains successfully**: With NaN handling in place
4. **Performance improvement**:
   - Accuracy: 41.6% → 48%+ (breakeven threshold)
   - ROI: -20.49% → Positive or near-positive
5. **Code maintainability**: Feature filtering system makes future changes easier

---

## References

- **Profitable Model**: RB rushing yards (31 features, positive ROI)
- **Current Implementation**: `modules/ml_feature_engineering.py`
- **Ensemble Models**: `modules/ml_ensemble.py`
- **Training Pipeline**: `modules/ml_training_data_builder.py`
- **Feature Docs**: Lines 14-26 in ml_feature_engineering.py

---

## Notes

- **Do not modify RB rushing model** - It's profitable, keep as-is
- **Use NaN for missing data** - Don't use hardcoded league averages
- **Match profitable patterns** - RB model uses 1 injury feature successfully
- **Tree models handle NaN** - XGBoost, LightGBM, CatBoost support native NaN
- **Test incrementally** - Rebuild training data after each phase to catch issues early
