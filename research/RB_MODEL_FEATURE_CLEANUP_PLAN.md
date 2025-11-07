# RB Model Feature Cleanup Plan

## Executive Summary

The enhanced RB rushing_yards model (51 features, 12,083 examples) performed **WORSE** than baseline:
- **Enhanced**: 39.5% accuracy, -24.50% ROI (220 test examples)
- **Baseline**: 40.2% accuracy, -23.24% ROI (291 test examples
- **Change**: -0.7% accuracy, -1.26% ROI

**Root Cause**: Circular and data-leaking features burying volume signal from 3 new rushing volume features.

**Solution**: Remove 3 problematic feature groups (13 features total) to let volume features shine:
1. **blocking_quality** (1 feature) - Circular logic
2. **Excessive injury features** (10 of 12 features) - Feature pollution
3. **team_rb_quality** (1 feature) - Data leakage for RB props

**Expected Outcome**: Features reduced from 51 → ~38-40, removing noise to expose carry share → yardage relationship.

---

## Problem Analysis

### Failed Enhancement Results

**Enhanced Model** (RB-only + 3 volume features):
- Training: 12,083 examples, 51 features
- Volume features: rushing_attempt_share_season, carry_share_3wk, goal_line_share
- Test accuracy: 39.5% (2024 only, 220 examples)
- ROI: -24.50%

**Baseline Model** (RB+QB mixed):
- Training: 17,207 examples (29% more data)
- Test accuracy: 40.2% (2024 only, 291 examples)
- ROI: -23.24%

**Selective Betting** (enhanced model, confidence thresholds):
- 20+ yard lines: 36.7% acc, -30.00% ROI (30 bets)
- 25+ yard lines: 37.5% acc, -28.41% ROI (16 bets)
- **No profitable threshold ranges found**

### Hypothesis: Feature Pollution

The enhanced model added 3 NEW volume features but got 51 total features (expected 48). Analysis revealed:

1. **Circular Features**: `blocking_quality` measures RB talent vs backup, not OL blocking
2. **Excessive Injury Features**: 12 injury features = 25% of model (likely overfitting on injury patterns vs yardage)
3. **Data Leakage**: `team_rb_quality` uses target player's performance to predict their performance
4. **Buried Signal**: 3 volume features (6% of model) buried under 13 noisy features (27% of model)

### User Correction: Weather Features Are Valuable

**My Initial Claim**: "Weather features (temp/wind) are NOISE for RBs - less affected than QBs/WRs"

**User Feedback**: "isnt weather features helpful because it informs the model that in these conditions, teams are more likely to run more or not?"

**User is RIGHT**:
- Bad weather (rain/wind/cold) → harder to pass → teams run MORE
- This is a **volume signal**, not efficiency
- QB model (51.7% acc) includes weather and works well
- **Decision**: KEEP weather features

---

## Feature Audit

### Current Features (51 total)

**Core Performance** (8):
- player_avg, last_3_avg, consistency_score, efficiency_score, home_road_split
- opp_rank_allowed, opp_season_avg_allowed, opp_rolling_avg_allowed

**Matchup** (4):
- matchup_history_avg, matchup_frequency, rival_game, division_game

**Vegas & Team** (6):
- vegas_total, vegas_spread, implied_team_total
- team_win_pct, team_offensive_rank, team_avg_score

**Weather** (4):
- temp, wind, weather_conditions (encoded), roof (encoded)

**Success Rate** (3):
- player_success_rate, league_avg_success_rate, opp_success_rate_allowed

**Game Script** (6):
- team_avg_margin, team_rb_quality, opp_def_ppg_allowed
- opp_def_ypg_allowed, team_plays_per_game, team_time_of_possession

**Injury** (12):
- injury_games_missed_y1, injury_games_missed_y2, injury_games_missed_y3
- injury_classification_score, has_recurring_injury
- games_missed_current_season, weeks_since_missed_game
- is_on_injury_report, injury_status_score
- injury_type_mobility, injury_type_upper_body

**Blocking Quality** (1):
- player_ypc, team_ypc, ypc_diff_pct

**Rushing Volume** (3 NEW):
- rushing_attempt_share_season
- carry_share_3wk
- goal_line_share

**Categoricals** (4):
- opponent (team abbr), position, week, season

**Total**: 51 features (expected 48 - 3 mystery features likely from blocking_quality method)

### Features to Remove (13 total)

#### 1. blocking_quality (1 feature)

**File**: `modules/ml_feature_engineering.py`
**Lines**: 162-166 (extraction), 591-620 (implementation)

**Problem**: Circular logic - measures RB talent vs backup, not OL blocking quality

**Implementation**:
```python
# Lines 591-620
def _extract_blocking_quality_features(...):
    # Get player YPC
    player_ypc = player_yards / player_carries

    # Get teammate RB YPC (EXCLUDING this player)
    team_ypc = teammate_yards / teammate_carries

    # Calculate "blocking quality" (CIRCULAR!)
    ypc_diff_pct = player_ypc / team_ypc
```

**Why This Is Circular**:
- Example: Derrick Henry (5.0 YPC) / Backup RB (3.5 YPC) = 1.43
- This measures "is this RB better than the backup?" not "is the OL blocking well?"
- If both RBs run behind the same OL, this is just measuring RB talent difference

**User's Intent** (correct): "inform whether teams will even bother rushing"
**Implementation** (wrong): Compares RB talent, not OL quality

**Solution**: Remove entirely (current implementation doesn't measure what's intended)

#### 2. Excessive Injury Features (10 of 12 features)

**File**: `modules/ml_feature_engineering.py`
**Lines**: 1038-1200 (implementation)

**Problem**: 12 injury features = 25% of model, likely overfitting on injury patterns vs yardage

**Features to REMOVE** (10):
- `injury_games_missed_y1` (historical)
- `injury_games_missed_y2` (historical)
- `injury_games_missed_y3` (historical)
- `injury_classification_score` (0=reliable, 1=moderate, 2=elevated, 3=injury-prone)
- `has_recurring_injury` (same body part 2+ times in 3 years)
- `games_missed_current_season` (season context)
- `weeks_since_missed_game` (recency)
- `injury_type_mobility` (ankle/knee/etc - affects running)
- `injury_type_upper_body` (shoulder/hand/etc - less relevant for RB)

**Features to KEEP** (2):
- `is_on_injury_report` (binary: 0 or 1) - Current week status
- `injury_status_score` (0=none, 1=questionable, 2=doubtful, 3=out) - Severity

**Rationale**:
- Current week injury status is MOST predictive
- Historical injury patterns likely noise (overfitting)
- Reduces features from 12 → 2 (10 features removed)

#### 3. team_rb_quality (1 feature) - Data Leakage for RB Props

**File**: `modules/ml_feature_engineering.py`
**Lines**: 1409+ (implementation in _extract_game_script_features)

**Problem**: For RB props, this feature uses target player's performance to predict their performance

**Implementation**:
```python
# Team RB Quality (YPC vs league average)
team_rb_ypc = team_rush_yards / team_rush_attempts
team_rb_quality = team_rb_ypc / league_avg_ypc
```

**Why This Is Data Leakage for RB Props**:
- Team RB YPC often IS the target player's YPC (primary back)
- Example: Derrick Henry carries 65% of team rushes
- team_rb_ypc ≈ Derrick Henry's YPC
- Using "is this RB good?" to predict "will this RB be good this week?" (circular)

**Why This Is Fine for QB/WR/TE Props**:
- QB passing props: team_rb_quality predicts game script (good run game → less passing)
- WR/TE receiving props: team_rb_quality predicts pass volume (bad run game → more passing)

**Solution**: Skip team_rb_quality for rushing_yards props only

**Implementation**:
```python
def _extract_game_script_features(..., prop_type):
    # ...

    # Team RB Quality (SKIP for rushing_yards to avoid data leakage)
    if prop_type != 'rushing_yards':
        # Calculate team_rb_quality for QB/WR/TE props
        features['team_rb_quality'] = team_rb_ypc / league_avg_ypc
    # else: omit feature for RB rushing props
```

### Features After Cleanup (~38-40 total)

**Removed**:
- 1 blocking_quality feature
- 10 excessive injury features
- 1 team_rb_quality (for RB props only)
- **Total removed**: 12-13 features

**Remaining**:
- 51 - 13 = **38 features** (approximate)

**Hypothesis**: Clean signal allows model to learn:
- carry_share_season × opponent_defense → baseline expectation
- carry_share_3wk vs carry_share_season → role change detection
- goal_line_share × team_red_zone_tendency → TD opportunity

---

## Implementation Steps

### Step 1: Modify Feature Extraction (modules/ml_feature_engineering.py)

#### Change 1: Comment Out blocking_quality Extraction

**Location**: Lines 162-166

**Current Code**:
```python
if position == 'RB':
    blocking_features = self._extract_blocking_quality_features(
        player_id, season, week
    )
    features.update(blocking_features)
```

**Modified Code**:
```python
# REMOVED: blocking_quality feature (circular logic - measures RB talent vs backup)
# if position == 'RB':
#     blocking_features = self._extract_blocking_quality_features(
#         player_id, season, week
#     )
#     features.update(blocking_features)
```

**Note**: The `_extract_blocking_quality_features()` method (lines 591-620) can remain in the file but will never be called.

---

#### Change 2: Simplify Injury Features to 2 Core Features

**Location**: Lines 1038-1200 (_extract_injury_features method)

**Current Code** (returns 12 features):
```python
def _extract_injury_features(
    self,
    player_id: str,
    season: int,
    week: int,
    position: str,
    prop_type: str
) -> Dict[str, float]:
    """
    Extract comprehensive injury-related features (12 features).
    """
    features = {}

    # 1. Historical injury pattern (3-year lookback)
    injury_history = []
    for year_offset in range(1, 4):
        # ... calculate injury_games_missed_y1/y2/y3

    features['injury_games_missed_y1'] = ...
    features['injury_games_missed_y2'] = ...
    features['injury_games_missed_y3'] = ...
    features['injury_classification_score'] = ...
    features['has_recurring_injury'] = ...

    # 2. Current season context
    features['games_missed_current_season'] = ...
    features['weeks_since_missed_game'] = ...

    # 3. Current week status (MOST IMPORTANT)
    features['is_on_injury_report'] = ...
    features['injury_status_score'] = ...
    features['injury_type_mobility'] = ...
    features['injury_type_upper_body'] = ...

    return features
```

**Modified Code** (returns 2 features):
```python
def _extract_injury_features(
    self,
    player_id: str,
    season: int,
    week: int,
    position: str,
    prop_type: str
) -> Dict[str, float]:
    """
    Extract current week injury status (2 features).

    SIMPLIFIED: Removed 10 historical/classification features (feature pollution).
    Keeps only current week injury status (most predictive signal).

    Features:
    - is_on_injury_report: Binary (0 or 1)
    - injury_status_score: Severity (0=none, 1=questionable, 2=doubtful, 3=out)
    """
    from modules.injury_cache_builder import load_injury_data

    features = {
        'is_on_injury_report': 0.0,
        'injury_status_score': 0.0
    }

    try:
        # Load injury data for current week
        injuries_df = load_injury_data(season)

        if not injuries_df.is_empty():
            # Look for injury report in week N (predicting for week N)
            player_injury = injuries_df.filter(
                (pl.col('gsis_id') == player_id) &
                (pl.col('week') == week)
            )

            if len(player_injury) > 0:
                features['is_on_injury_report'] = 1.0

                # Status severity (0=none, 1=questionable, 2=doubtful, 3=out)
                status = player_injury['report_status'][0]
                status_map = {'Questionable': 1, 'Doubtful': 2, 'Out': 3}
                features['injury_status_score'] = float(status_map.get(status, 1))

    except Exception as e:
        logger.debug(f"Error extracting injury features: {e}")

    return features
```

**Features Removed**:
- injury_games_missed_y1/y2/y3 (historical)
- injury_classification_score (pattern classification)
- has_recurring_injury (recurring body part)
- games_missed_current_season (season context)
- weeks_since_missed_game (recency)
- injury_type_mobility (ankle/knee injuries)
- injury_type_upper_body (shoulder/hand injuries)

**Features Kept**:
- is_on_injury_report (binary: on report this week?)
- injury_status_score (severity: questionable=1, doubtful=2, out=3)

---

#### Change 3: Skip team_rb_quality for RB Props

**Location**: Lines 1329+ (_extract_game_script_features method)

**Current Code**:
```python
def _extract_game_script_features(
    self,
    player_id: str,
    season: int,
    week: int,
    position: str,
    opponent_team: str
) -> Dict[str, float]:
    """
    Extract game script and team context features (6 features).
    """
    features = {
        'team_avg_margin': 0.0,
        'team_rb_quality': 1.0,  # Always initialized
        'opp_def_ppg_allowed': 22.0,
        'opp_def_ypg_allowed': 340.0,
        'team_plays_per_game': 65.0,
        'team_time_of_possession': 30.0
    }

    # ... load data ...

    # Team RB Quality (YPC vs league average)
    team_rush_attempts = pbp_filtered.filter(...)
    team_rb_ypc = team_rush_yards / len(team_rush_attempts)
    league_avg_ypc = 4.3  # Approximate
    features['team_rb_quality'] = team_rb_ypc / league_avg_ypc

    return features
```

**Modified Code**:
```python
def _extract_game_script_features(
    self,
    player_id: str,
    season: int,
    week: int,
    position: str,
    opponent_team: str,
    prop_type: str  # ADD prop_type parameter
) -> Dict[str, float]:
    """
    Extract game script and team context features (5-6 features).

    NOTE: team_rb_quality is skipped for rushing_yards props to avoid
    data leakage (team RB YPC often IS the target player's YPC).
    """
    features = {
        'team_avg_margin': 0.0,
        'opp_def_ppg_allowed': 22.0,
        'opp_def_ypg_allowed': 340.0,
        'team_plays_per_game': 65.0,
        'team_time_of_possession': 30.0
    }

    # Conditionally initialize team_rb_quality
    if prop_type != 'rushing_yards':
        features['team_rb_quality'] = 1.0

    # ... load data ...

    # Team RB Quality (SKIP for rushing_yards to avoid data leakage)
    if prop_type != 'rushing_yards':
        team_rush_attempts = pbp_filtered.filter(...)
        team_rb_ypc = team_rush_yards / len(team_rush_attempts)
        league_avg_ypc = 4.3
        features['team_rb_quality'] = team_rb_ypc / league_avg_ypc

    return features
```

**Important**: Update the method call in `generate_features()` (around line 180):
```python
# OLD:
game_script_features = self._extract_game_script_features(
    player_id, season, week, position, opponent_team
)

# NEW:
game_script_features = self._extract_game_script_features(
    player_id, season, week, position, opponent_team, prop_type
)
```

---

### Step 2: Rebuild Training Data

**Command**:
```bash
python -c "
from modules.ml_training_data_builder import TrainingDataBuilder
import logging

logging.basicConfig(level=logging.INFO)

print('='*70)
print('REBUILDING CLEANED RB TRAINING DATA')
print('='*70)
print('Prop Type: rushing_yards (RB-only)')
print('Years: 2015-2024')
print('Expected Features: ~38-40 (down from 51)')
print()
print('REMOVED Features (13):')
print('1. blocking_quality (1 feature) - Circular logic')
print('2. Injury features (10 of 12) - Feature pollution')
print('3. team_rb_quality (1 feature) - Data leakage for RB props')
print()
print('KEPT Features:')
print('- 3 volume features (rushing_attempt_share, carry_share_3wk, goal_line_share)')
print('- Weather features (volume predictors for run game)')
print('- 2 core injury features (is_on_injury_report, injury_status_score)')
print()

builder = TrainingDataBuilder()
df = builder.build_training_dataset(
    prop_type='rushing_yards',
    start_year=2015,
    end_year=2024
)

print(f'\n{'='*70}')
print(f'Training data built successfully!')
print(f'Total examples: {len(df):,}')
print(f'Total features: {len(df.columns)}')
print(f'Expected: ~38-40 features')
print(f'Status: {\"✅ CLEAN\" if 38 <= len(df.columns) <= 40 else f\"⚠️  Got {len(df.columns)}\"}')
print(f'Saved to: cache/ml_training_data/rushing_yards_2015_2024.parquet')
print(f'{'='*70}')
"
```

**Expected Output**:
```
Training data built successfully!
Total examples: 12,083
Total features: 38-40
Status: ✅ CLEAN
```

---

### Step 3: Train Cleaned Model

**Command**:
```bash
python scripts/train_all_prop_types.py --props rushing_yards --start-year 2015 --end-year 2024
```

**Expected Output**:
```
Training rushing_yards model...
Examples: 12,083
Features: 38-40
Ensemble: XGBoost + LightGBM + CatBoost + RandomForest
Saved to: models/rushing_yards_ensemble.pkl
```

---

### Step 4: Evaluate Cleaned Model

**Command**:
```bash
python scripts/evaluate_prop_type.py rushing_yards
```

**Compare Against Baseline**:
- **Baseline**: 40.2% acc, -23.24% ROI (291 test examples)
- **Enhanced (failed)**: 39.5% acc, -24.50% ROI (220 test examples)
- **Cleaned (target)**: ? acc, ? ROI

**Success Criteria**:
- Accuracy ≥ 40.2% (match or beat baseline)
- ROI improvement vs enhanced model
- Profitable threshold range found (52.4%+ accuracy at some confidence level)

---

## Expected Outcome

### Feature Count Reduction

**Before Cleanup**: 51 features
- 45 baseline
- 3 volume (NEW)
- 3 mystery (from blocking_quality method)

**After Cleanup**: ~38-40 features
- 45 baseline
- 3 volume (NEW)
- **-1** blocking_quality
- **-10** excessive injury features
- **-1** team_rb_quality (for RB props)

**Net**: 51 - 13 = **38 features**

### Hypothesis Testing

**Hypothesis**: Clean signal allows volume features to shine

**Mechanism**:
1. **carry_share_season** (baseline role): "Is this a bellcow (60%+) or committee back (30%)?"
2. **carry_share_3wk** (recent role): "Has role changed? Breakout? Injury return? Losing snaps?"
3. **goal_line_share** (high-value touches): "Does this RB get goal-line carries?"

**Model Learning Path**:
- High carry share × weak opponent defense → high yardage expectation
- carry_share_3wk >> carry_share_season → breakout signal (increase prediction)
- carry_share_3wk << carry_share_season → losing touches (decrease prediction)
- High goal_line_share × team_red_zone_tendency → more TD opportunities

**Why Cleanup Helps**:
- Removes circular features (blocking_quality) that confuse the model
- Removes leaky features (team_rb_quality) that create false confidence
- Removes excessive injury features (10 features) burying volume signal
- Lets 3 volume features (6% of clean model) learn carry share → yardage relationship

---

## Fallback Plan

**If cleaned model still fails (< 40.2% accuracy)**:

### Option 1: Restore QB Rushing Data

**Problem**: Lost 29% of training data (17,207 → 12,083) from RB-only filter

**Solution**: Restore QB rushing data but add position-specific volume features
- QB volume: designed_rush_rate, scramble_rate, red_zone_rush_tendency
- RB volume: carry_share_season, carry_share_3wk, goal_line_share
- Model learns position-specific patterns

### Option 2: Accept Fundamental Difficulty

**Reality Check**:
- Passing yards: 51.7% acc (40 attempts, low variance per attempt)
- Receiving yards: 41.6% acc (target share is STRONG signal)
- Rushing yards: 40.2% acc (20 carries, high variance per carry)

**Why RB Rushing Is Harder**:
1. **Fewer opportunities**: 20 carries vs 40 pass attempts
2. **Higher variance**: RB can get stuffed 8 times then break an 80-yard TD
3. **Game script dominance**: Trailing teams abandon run (unpredictable)
4. **Carry share is weaker than target share**: Target share → receptions (direct), carry share → yards (indirect through efficiency)

**Acceptance Criteria**:
- If cleaned model reaches 40-41% accuracy, consider it near the prediction ceiling for RB rushing
- Focus on improving other props (WR receiving, QB passing) where signal is stronger

---

## Files Modified

### modules/ml_feature_engineering.py
**Lines Modified**:
- 162-166: Comment out blocking_quality extraction
- 180-183: Add prop_type parameter to _extract_game_script_features call
- 1038-1200: Simplify _extract_injury_features to 2 core features
- 1329+: Add prop_type parameter, skip team_rb_quality for rushing_yards

### cache/ml_training_data/rushing_yards_2015_2024.parquet
**Rebuilt**: ~38-40 features (down from 51)

### models/rushing_yards_ensemble.pkl
**Retrained**: New model on cleaned training data

---

## Success Metrics

### Primary Metrics
- **Accuracy**: ≥ 40.2% (match or beat baseline)
- **ROI**: Improved vs enhanced model (-24.50%)
- **Feature Count**: ~38-40 (cleaned from 51)

### Secondary Metrics
- **Profitable Threshold**: Find confidence level with 52.4%+ accuracy
- **Sample Size**: Maintain ~12,000 training examples (RB-only)
- **Feature Interpretability**: Volume features should have high importance

### Validation
- Test on 2023-2024 data with real betting lines
- Compare selective betting ranges (20+ yards, 25+ yards)
- Check feature importance rankings (volume features should be top 10)

---

## References

### Related Files
- [modules/ml_feature_engineering.py](modules/ml_feature_engineering.py) - Feature extraction (TO BE MODIFIED)
- [modules/ml_training_data_builder.py](modules/ml_training_data_builder.py) - Training data builder
- [modules/prop_types.py](modules/prop_types.py) - Prop type configuration
- [scripts/train_all_prop_types.py](scripts/train_all_prop_types.py) - Model training script
- [scripts/evaluate_prop_type.py](scripts/evaluate_prop_type.py) - Model evaluation script

### User Feedback
- Weather features: "isnt weather features helpful because it informs the model that in these conditions, teams are more likely to run more or not?" ✅ CORRECT
- Blocking quality intent: "isnt blocking quality good because itll inform whether teams will evevn bother rushing?" ✅ INTENT CORRECT, IMPLEMENTATION WRONG

### Baseline Results
- **Baseline** (RB+QB mixed): 40.2% acc, -23.24% ROI (291 examples)
- **Enhanced** (RB-only + volume): 39.5% acc, -24.50% ROI (220 examples)
- **Target** (cleaned features): TBD

---

## Timeline

1. **Step 1**: Modify ml_feature_engineering.py (15 minutes)
2. **Step 2**: Rebuild training data (30-45 minutes)
3. **Step 3**: Train cleaned model (20-30 minutes)
4. **Step 4**: Evaluate and report results (10 minutes)

**Total**: ~75-100 minutes

---

## Notes

- Keep weather features (user confirmed they predict run volume in adverse conditions)
- Remove blocking_quality entirely (circular implementation doesn't measure intended signal)
- Simplify injury features to 2 core (current week status only)
- Skip team_rb_quality for RB props only (fine for QB/WR/TE props)
- This is a surgical cleanup targeting 3 specific feature groups (13 features total)
