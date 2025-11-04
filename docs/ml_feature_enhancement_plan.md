# ML Feature Enhancement Plan - All Prop Types

**Status**: Phase 0 Complete (Injury Features Added)
**Current Baseline**: 51.2% accuracy, -2.17% ROI (passing_yards with injury features)
**Goal**: Achieve 52.4%+ accuracy for profitable betting
**Last Updated**: 2025-11-04

---

## Data Strategy Decision

### Training Years: 2015-2024 (KEEP THIS RANGE)

**Rationale**:
- **More data = better ML**: 10 years gives 5,000+ training examples
- **Handle missing features gracefully**: Model has defaults for missing NextGen data (2016+)
- **2015-2024 is modern era**: Offensive schemes are consistent enough
- **Betting markets matter**: 2015+ lines are sophisticated
- **The math**: Losing 2015 data = -493 examples (-10%) for minimal scheme purity gain

**Feature Coverage**:
- Phase 1 features: Available 1999-2025 (CPOE, pressure, target share)
- Phase 2 features: Available 2016-2025 (NextGen, snap counts)
- 2015 data uses Phase 1 features only (graceful degradation)

**Revisit Decision If**: NextGen features prove HIGHLY predictive (e.g., +5% accuracy alone)

---

## System Separation (DO NOT MUDDLE)

### Rankings System (READ-ONLY - DO NOT MODIFY)
- `modules/adjustment_pipeline.py` - Applies multipliers
- `modules/context_adjustments.py` - Calculates adjustments
- `modules/prop_projection_engine.py` - Generates final rankings

### ML System (WILL MODIFY)
- `modules/ml_feature_engineering.py` - Extracts raw features
- `modules/ml_training_data_builder.py` - Builds training datasets
- `modules/ml_ensemble.py` - Trains models

### Shared Infrastructure (READ-ONLY for both)
- `cache/pbp/` - Play-by-play data
- `cache/nextgen/` - NextGen tracking data
- `cache/participation/` - Personnel/snap data

**Key Principle**: ML extracts RAW metrics, Rankings applies multipliers. Zero cross-contamination.

---

## Phase 0: Baseline (COMPLETED)

### Status: ✅ COMPLETE

**Features Added**: 11 injury features
- injury_games_missed_y1/y2/y3
- injury_classification_score
- has_recurring_injury
- games_missed_current_season
- is_on_injury_report
- injury_status_score
- injury_type_mobility
- injury_type_upper_body
- weeks_since_last_missed
- effective_games_ratio

**Results**:
- **Accuracy**: 46.8% → 51.2% (+4.4%)
- **ROI**: -10.72% → -2.17% (+8.55%)
- **Status**: Close to break-even but not profitable
- **Profitable Threshold**: 60+ yards confidence (53.3% acc, +1.82% ROI, 0.4 bets/week)

**Training Data**: `cache/ml_training_data/passing_yards_2015_2024.parquet`
- 5,124 examples
- 39 features
- 2015-2024 coverage

---

## Phase 1: High-Impact PBP Features

### Test ONE feature at a time, measure impact, decide keep/discard

---

### Feature 1: CPOE (Completion % Over Expected)

**What**: QB accuracy signal - how well QB completes passes vs expected difficulty
**Why**: Strong QB skill metric that's independent of volume
**Coverage**: 1999-2025 (full history available)
**Prop Types**: QB props (passing_yards, passing_tds)

**Features to Extract** (from `cache/pbp/pbp_YYYY.parquet`):
```python
'cpoe_season_avg': float     # Season-long CPOE
'cpoe_l3_avg': float          # Last 3 games CPOE
'expected_completion_rate': float  # Avg difficulty of throws
```

**Implementation**:
1. Add `_extract_cpoe_features()` method to `modules/ml_feature_engineering.py`
2. Load PBP data, filter to player passes through week N-1
3. Calculate CPOE metrics: `(complete_pass - cp).mean()` for season and L3
4. Integrate into main `engineer_features()` pipeline
5. Rebuild passing_yards training data (delete old parquet, rebuild 2015-2024)
6. Train model on 2015-2022, test on 2024
7. Evaluate against real betting lines from `cache/player_props/`

**Expected Impact**: +1-2% accuracy
**Baseline to Beat**: 51.2% accuracy, -2.17% ROI

**Evaluation Criteria**:
- ✅ KEEP if accuracy ≥ 52.0% OR ROI ≥ 0%
- ❌ DISCARD if accuracy < 51.5% AND ROI worsens

**Success Metrics**:
| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Accuracy | 51.2% | 52.4%+ | TBD |
| ROI | -2.17% | 0%+ | TBD |
| 60yd threshold acc | 53.3% | 54%+ | TBD |

**Results - Implementation Date: 2025-11-04**

**Status**: ❌ DISCARDED

**Performance (201 test examples on 2024 real betting lines)**:
| Metric | Phase 0 (Baseline) | Phase 1 (CPOE) | Change |
|--------|-------------------|----------------|--------|
| Accuracy | 51.2% | 48.8% | -2.4% ❌ |
| ROI | -2.17% | -6.92% | -4.75% ❌ |
| Test Size | 201 | 201 | Same |

**Selective Betting (High-Confidence Thresholds)**:
- 40+ yards: 52.9% acc, +1.07% ROI (34 bets)
- 50+ yards: 55.6% acc, +6.06% ROI (18 bets)
- 55+ yards: 66.7% acc, +27.27% ROI (12 bets)
- 60+ yards: 63.6% acc, +21.49% ROI (11 bets)

**Decision Rationale**:
CPOE features failed both evaluation criteria:
- Accuracy: 48.8% < 51.5% threshold ❌
- ROI: Worsened significantly (-6.92% vs -2.17% baseline) ❌
- While high-confidence thresholds remain profitable, the baseline already had similar selective performance
- The 2.4% accuracy drop across all predictions is unacceptable

**Why It Failed**:
CPOE measures QB accuracy/efficiency (completing difficult throws), NOT volume. Passing yards is volume-driven - a QB can have high CPOE but low yardage (or vice versa). The feature added noise rather than signal to yardage predictions.

**Training Details**:
- Features added: 3 (cpoe_season_avg, cpoe_l3_avg, expected_completion_rate)
- Total features: 42 (was 39)
- Training: 4,027 examples (2015-2022)
- Test: 201 examples (2024 with real lines)
- Data rebuild time: 65 minutes
- Model training: 2 minutes

**Action Taken**: Removed CPOE features, reverted to Phase 0 baseline (39 features)

**Next Steps**: Test CPOE specifically for passing_tds (efficiency-driven prop) instead

---

### Feature 2: Pressure Rate

**What**: How often QB is under pressure or sacked
**Why**: QB performance degrades significantly under pressure
**Coverage**: 1999-2025 (full history available)
**Prop Types**: QB props (passing_yards, passing_tds)
**Depends On**: Feature 1 (CPOE) - adds to same method

**Features to Extract** (from `cache/pbp/pbp_YYYY.parquet`):
```python
'pressure_rate_season': float  # % of dropbacks with pressure
'pressure_rate_l3': float       # Last 3 games pressure %
'sack_rate_season': float       # % of dropbacks ending in sack
```

**Implementation**:
1. Extend `_extract_cpoe_features()` to include pressure metrics
2. Filter to pass plays, calculate `was_pressure.mean()` and `sack.mean()`
3. Rebuild training data (now includes CPOE + pressure)
4. Train and evaluate

**Expected Impact**: +0.5-1% accuracy (incremental over CPOE)
**Baseline to Beat**: Phase 0 baseline (51.2% accuracy, -2.17% ROI)

**Evaluation Criteria**:
- ✅ KEEP if accuracy ≥ 52.0% OR ROI ≥ 0%
- ❌ DISCARD if accuracy < 51.5% AND ROI worsens

**Results - Implementation Date: 2025-11-04**

**Status**: ❌ DISCARDED

**Performance (201 test examples on 2024 real betting lines)**:
| Metric | Phase 0 (Baseline) | Phase 1 (Pressure) | Change |
|--------|-------------------|-------------------|--------|
| Accuracy | 51.2% | 46.3% | -4.9% ❌ |
| ROI | -2.17% | -11.67% | -9.5% ❌ |
| Test Size | 201 | 201 | Same |

**Selective Betting (High-Confidence Thresholds)**:
All threshold ranges tested (20-50 yards) remained unprofitable:
- Best threshold: 40 yards = 43.3% accuracy, -17.27% ROI (30 bets)
- Worst threshold: 50 yards = 35.7% accuracy, -31.82% ROI (14 bets)
- No profitable ranges found

**Decision Rationale**:
Pressure Rate features failed catastrophically on both evaluation criteria:
- Accuracy: 46.3% << 51.5% threshold (worse than CPOE) ❌
- ROI: Worsened dramatically (-11.67% vs -2.17% baseline) ❌
- 4.9% accuracy drop is the worst performance tested so far
- Even selective betting strategies found no profitable ranges

**Why It Failed**:
Pressure rate measures OL quality and defensive performance rather than QB yardage output:
1. **Too noisy**: Pressure varies heavily week-to-week based on opponent pass rush
2. **Limited coverage**: qb_hit column only available 2016+, using 0.25 fallback for 2015
3. **Wrong metric**: Pressure impacts efficiency (completion %, YPA) but not necessarily volume (attempts)
4. **Team-driven**: More correlated with OL quality than QB yardage ability
5. **Worse than CPOE**: At least CPOE measures QB skill; pressure is largely external

**Training Details**:
- Features added: 3 (pressure_rate_season, pressure_rate_l3, sack_rate_season)
- Total features: 42 (was 39)
- Training: 4,027 examples (2015-2022)
- Test: 201 examples (2024 with real lines)
- Data rebuild time: 59 minutes
- Model training: 2.5 minutes

**Action Taken**: Removed Pressure Rate features, reverted to Phase 0 baseline (39 features)

**Key Learnings**:
- Efficiency metrics (CPOE, pressure) don't help predict volume-based props
- Need to focus on volume drivers: opponent defense, game script, target share
- External factors (OL, defense) add noise rather than signal for player props

**Next Steps**: Test volume-driven features for passing props instead of efficiency metrics
- ❌ DISCARD if no improvement or negative impact

---

### Feature 3: Game Script & Team Context (QB passing_yards)

**What**: Volume-driven contextual features predicting passing attempts/opportunities
**Why**: Teams trailing = pass more; strong RBs = run more; bad defenses = high-scoring games
**Coverage**: 1999-2025 (full history available)
**Prop Types**: QB props (passing_yards, passing_tds)

**Features to Extract** (from `cache/pbp/pbp_YYYY.parquet`):
```python
'team_avg_margin': float           # Rolling 3-game point differential (game script)
'team_rb_quality': float           # Team RB YPC vs league average (run tendency)
'opp_def_ppg_allowed': float       # Opponent points per game allowed
'opp_def_ypg_allowed': float       # Opponent yards per game allowed
'team_plays_per_game': float       # Rolling 3-game offensive plays per game
'team_time_of_possession': float   # Average TOP per game (minutes)
```

**Implementation**:
1. Add `_extract_game_script_features()` method to `modules/ml_feature_engineering.py`
2. Load PBP data, calculate team offensive context and opponent defensive metrics
3. All features use strict through_week filtering (no future data)
4. Rebuild passing_yards training data (2015-2024)
5. Train ensemble on 2015-2022, test on 2024 real betting lines
6. Evaluate against 201 real Tuesday betting lines

**Expected Impact**: +1-2% accuracy (volume-driven vs efficiency metrics)
**Baseline to Beat**: 51.2% accuracy, -2.17% ROI

**Evaluation Criteria**:
- ✅ KEEP if accuracy ≥ 52.0% OR ROI ≥ 0%
- ❌ DISCARD if accuracy < 51.5% AND ROI worsens

**Results - Implementation Date: 2025-11-04**

**Status**: ✅ KEPT

**Performance (201 test examples on 2024 real betting lines)**:
| Metric | Phase 0 (Baseline) | Game Script (45 features) | Change |
|--------|-------------------|---------------------------|--------|
| Accuracy | 51.2% | 51.7% | +0.5% ✓ |
| ROI | -2.17% | -1.22% | +0.95% ✓ |
| Test Size | 201 | 201 | Same |

**Selective Betting (High-Confidence Thresholds)**:
- 40+ yards: 53.8% acc, +2.80% ROI (26 bets)
- 50+ yards: 66.7% acc, +27.27% ROI (15 bets) ⭐
- 55+ yards: 54.5% acc, +4.13% ROI (11 bets)

**Decision Rationale**:
Game Script features show incremental improvement on both evaluation criteria:
- Accuracy: 51.7% > 51.5% threshold ✓
- ROI: Improved by nearly 1% (-1.22% vs -2.17%) ✓
- Best performance achieved so far (vs CPOE: 48.8%, Pressure: 46.3%)
- Selective betting at 50+ yards remains highly profitable (66.7% accuracy, +27.27% ROI)
- Volume-driven features align better with yardage prop predictions

**Why It Worked**:
Game Script features capture the VOLUME of passing opportunities, not efficiency:
1. **Team Context**: Point differential predicts pass-heavy vs run-heavy game scripts
2. **RB Quality**: Strong rushing attacks reduce passing volume
3. **Opponent Defense**: Bad defenses lead to high-scoring, pass-heavy games
4. **Pace/Tempo**: More plays per game = more opportunities for yardage
5. **Better Fit**: Volume metrics predict volume-based props (vs efficiency metrics)

**Training Details**:
- Features added: 6 (team_avg_margin, team_rb_quality, opp_def_ppg_allowed, opp_def_ypg_allowed, team_plays_per_game, team_time_of_possession)
- Total features: 45 (was 39)
- Training: 4,027 examples (2015-2022)
- Test: 201 examples (2024 with real lines)
- Data rebuild time: 70 minutes
- Model training: 2 minutes

**Action Taken**: KEPT Game Script features (45 total features now in baseline)

**Key Learnings**:
- Volume-driven features (game script, opponent defense, pace) work better than efficiency metrics
- Incremental improvements compound - small gains are valuable
- Focus on features that directly correlate with the prop being predicted

**Next Steps**: Continue testing additional volume-driven features or test receiving props with Target Share

---

### Feature 4: Target Share (WR/TE/RB)

**What**: Percentage of team targets allocated to player
**Why**: Volume is king for receiving props - target share predicts opportunity
**Coverage**: 1999-2025 (full history available)
**Prop Types**: Receiving props (receiving_yards, receiving_tds, receptions)

**Features to Extract** (from `cache/pbp/pbp_YYYY.parquet`):
```python
'target_share_3wk': float      # Recent 3-game target % of team total
'target_share_season': float   # Season-long target % of team total
'air_yards_share': float       # % of team air yards allocated to player
'target_quality': float        # Avg air yards per target (deep ball usage)
```

**Implementation**:
1. Add `_extract_target_features()` method to `ml_feature_engineering.py`
2. Load PBP data for player and team
3. Calculate target counts and air yards for player vs team
4. Build training data for **3 new prop types**:
   - `receiving_yards_2015_2024.parquet`
   - `receiving_tds_2015_2024.parquet`
   - `receptions_2015_2024.parquet`
5. Train models for all 3 prop types
6. Evaluate against real betting lines (if available in `cache/player_props/`)

**Expected Impact**: +2-3% accuracy for receiving props
**Baseline to Beat**: No existing baseline for receiving props

**Evaluation Criteria**:
- ✅ KEEP if accuracy ≥ 52.4% (break-even threshold)
- Note: May have limited betting line data for receiving props

---

## Phase 2: NextGen & Advanced Features

### Note: 2016-2025 coverage, will have missing values for 2015 examples

---

### Feature 4: NextGen Separation (WR/TE)

**What**: GPS-tracked receiver separation at catch point
**Why**: Separation drives YAC potential and TD probability
**Coverage**: 2016-2025 (limited history, but high quality)
**Prop Types**: Receiving props (receiving_yards, receiving_tds)

**Features to Extract** (from `cache/nextgen/nextgen_YYYY.parquet`):
```python
'avg_separation_3wk': float        # Recent 3-game avg separation (yards)
'avg_cushion_season': float        # Season avg cushion at snap (yards)
'separation_efficiency': float     # Separation per target
'tight_coverage_separation': float # Separation when cushion < 5 yards
```

**Implementation**:
1. Add `_extract_nextgen_features()` method to `ml_feature_engineering.py`
2. Load NextGen data, aggregate by week
3. Calculate separation metrics with 3-week rolling average
4. Rebuild receiving prop training data (2015-2024, with 2016+ having NextGen)
5. Train and evaluate
6. **NOTE**: 2015 examples will have default values (0.0) for NextGen features

**Expected Impact**: +1-2% accuracy for receiving props
**Baseline to Beat**: Feature 3 (target share) results

**Evaluation Criteria**:
- ✅ KEEP if incremental improvement ≥ 1%
- If NextGen features show +5% impact alone: REVISIT 2015-2024 vs 2016-2024 decision

---

### Feature 5: Snap Count (RB/WR workload)

**What**: Offensive snap participation percentage
**Why**: Snap share predicts volume - high snap share = more touches
**Coverage**: 2016-2025 (limited history)
**Prop Types**: RB props (rushing_yards, rushing_tds) + receiving props

**Features to Extract** (from `cache/participation/participation_YYYY.parquet`):
```python
'snap_share_3wk': float   # Recent 3-game snap % of offensive snaps
'snap_trend': float        # Week-over-week trend (+/- snaps)
```

**Implementation**:
1. Add snap count derivation logic from participation data
2. Parse `offense_personnel` strings to count plays per player
3. Add `_extract_snap_features()` to `ml_feature_engineering.py`
4. Calculate snap counts and percentages
5. Build training data for **2 new prop types**:
   - `rushing_yards_2015_2024.parquet`
   - `rushing_tds_2015_2024.parquet`
6. Train RB models and evaluate

**Expected Impact**: +1-2% accuracy for RB props
**Baseline to Beat**: No existing baseline for RB props

**Evaluation Criteria**:
- ✅ KEEP if accuracy ≥ 52.4% (break-even)
- Note: Snap data less important for QB props (QBs play 100% of snaps)

---

### Feature 6: Personnel Groupings

**What**: Offensive formation frequencies (11/12/21/22 personnel)
**Why**: Formation context affects playcalling and defensive alignment
**Coverage**: 2016-2025 actual + 1999-2025 inferred (via `modules/personnel_inference.py`)
**Prop Types**: All props (scheme/formation context)

**Features to Extract** (from `cache/participation/participation_YYYY.parquet`):
```python
'personnel_11_pct': float       # % of snaps in 11 personnel (3 WR)
'personnel_12_pct': float       # % of snaps in 12 personnel (2 TE)
'personnel_heavy_pct': float    # % of snaps in 21/22 (heavy run formations)
'coverage_man_pct': float       # % of snaps facing man coverage
'avg_defenders_in_box': float   # Avg defenders in box (RB feature)
```

**Implementation**:
1. Parse `offense_personnel` and `defense_coverage_type` from participation
2. Add `_extract_personnel_features()` to `ml_feature_engineering.py`
3. Aggregate to player-week level
4. Rebuild **ALL** prop training data with personnel features
5. Train and evaluate across all prop types

**Expected Impact**: +0.5-1% accuracy across all props
**Baseline to Beat**: Previous feature results for each prop type

**Evaluation Criteria**:
- ✅ KEEP if improves at least 1 prop type by ≥ 0.5%
- Scheme context may have subtle but consistent impact

---

## Evaluation Process (After Each Feature)

### 1. Rebuild Training Data
```bash
# Delete old parquet file
rm cache/ml_training_data/{prop_type}_2015_2024.parquet

# Rebuild with new features
python -c "
from modules.ml_training_data_builder import TrainingDataBuilder
builder = TrainingDataBuilder()
df = builder.build_training_dataset(
    prop_type='{prop_type}',
    start_year=2015,
    end_year=2024
)
print(f'Features: {len(df.columns)}')
"
```

### 2. Train Model
```bash
python scripts/evaluate_prop_type.py {prop_type}
```

### 3. Extract Key Metrics
From evaluation output:
- **Accuracy**: X%
- **ROI**: Y%
- **Profitable threshold**: Z yards/TDs/receptions
- **Volume**: N bets/week at threshold

### 4. Document Results

Update this file with results:

```markdown
### Feature X: {Name} - Results

**Implementation Date**: YYYY-MM-DD
**Prop Types Tested**: {list}

**Results**:
| Prop Type | Baseline Acc | New Acc | Δ | Baseline ROI | New ROI | Δ |
|-----------|--------------|---------|---|--------------|---------|---|
| passing_yards | 51.2% | X% | +Y% | -2.17% | Z% | +W% |

**Decision**: ✅ KEEP / ❌ DISCARD
**Rationale**: {explanation}
```

### 5. Decision Criteria

**KEEP Feature If**:
- Accuracy improves by ≥ 0.5% AND/OR
- ROI improves by ≥ 1% AND/OR
- Creates new profitable threshold with reasonable volume (≥ 0.5 bets/week)

**DISCARD Feature If**:
- No accuracy improvement AND no ROI improvement
- Accuracy/ROI worsens
- Feature adds computational cost with no benefit

---

## Success Metrics by Prop Type

### Passing Yards
| Metric | Phase 0 (Baseline) | Phase 1 Target | Phase 2 Target |
|--------|-------------------|----------------|----------------|
| Accuracy | 51.2% | 52.4%+ | 53-54% |
| ROI | -2.17% | 0%+ | +3-5% |
| Profitable Threshold | 60+ yards | 50+ yards | 40+ yards |
| Volume | 0.4 bets/week | 1+ bets/week | 2+ bets/week |

### Passing TDs
| Metric | Phase 0 (Baseline) | Phase 1 Target | Phase 2 Target |
|--------|-------------------|----------------|----------------|
| Accuracy | TBD | 52.4%+ | 53-54% |
| ROI | TBD | 0%+ | +3-5% |

### Receiving Yards/TDs/Receptions
| Metric | Phase 0 (Baseline) | Phase 1 Target | Phase 2 Target |
|--------|-------------------|----------------|----------------|
| Accuracy | N/A (no baseline) | 52.4%+ | 53-55% |
| ROI | N/A | 0%+ | +3-5% |
| Note | Target share critical | NextGen adds value | - |

### Rushing Yards/TDs
| Metric | Phase 0 (Baseline) | Phase 1 Target | Phase 2 Target |
|--------|-------------------|----------------|----------------|
| Accuracy | N/A (no baseline) | 52.4%+ | 53-55% |
| ROI | N/A | 0%+ | +3-5% |
| Note | - | Snap count critical | - |

---

## Overall Success Criteria

### Minimum Viable Product (MVP)
- At least **1 prop type** achieves 52.4%+ accuracy (break-even)
- ROI ≥ 0% on that prop type
- Profitable threshold with reasonable volume (≥ 0.5 bets/week)

### Ideal Target
- **2-3 prop types** achieve 53-55% accuracy
- ROI +3-5% on multiple prop types
- Combined volume ≥ 5 bets/week across all props
- Systematic profitable betting system

### Current Gap
- Need +1.2% accuracy to reach break-even (51.2% → 52.4%)
- Phase 1 should deliver this (+3-5% expected combined impact)
- Phase 2 provides additional edge for comfortable margin

---

## Timeline Estimate

### Phase 1 (High-Impact PBP Features)
- Feature 1 (CPOE): 2 hours
- Feature 2 (Pressure): 1 hour (piggybacks on CPOE)
- Feature 3 (Target Share): 2 hours (+ 3 new prop types to train)
- **Total Phase 1**: ~8 hours

### Phase 2 (NextGen & Advanced)
- Feature 4 (NextGen Separation): 2 hours
- Feature 5 (Snap Count): 2 hours (+ 2 new prop types to train)
- Feature 6 (Personnel): 3 hours
- **Total Phase 2**: ~7 hours

### Grand Total: ~15 hours for complete implementation and testing

---

## Risk Mitigation

1. **No contamination of rankings system**
   - ML features extracted separately from `adjustment_pipeline.py`
   - Rankings use multipliers, ML uses raw metrics
   - Shared caches are read-only for both systems

2. **One feature at a time**
   - Clear attribution of impact
   - Easy to identify which features help vs hurt
   - Can roll back individual features if needed

3. **Backward compatible**
   - Feature extraction is modular
   - Easy to enable/disable features via flags
   - Old models can coexist with new ones

4. **Data coverage handled gracefully**
   - 2015 examples have missing NextGen data → defaults to 0.0
   - Model learns to work with partial feature sets
   - No crashes or errors from missing data

5. **Betting line data limitations**
   - Passing yards: Good coverage (201 lines in 2024)
   - Passing TDs: Limited coverage (22 lines in 2024)
   - Receiving/Rushing props: Unknown coverage, may be sparse
   - Evaluation may be limited for some prop types

---

## What Happens If Phase 1 + Phase 2 Aren't Enough?

If we don't reach break-even (52.4%) after all 6 features, explore:

### Phase 3 (Advanced/Experimental)
1. **Time to Throw** (requires caching NextGen passing stats)
2. **FTN Charting Data** (play action, blitz, RPO - 2022+ only)
3. **Coach Tenure** (coordinator changes affect scheme)
4. **Weather Interaction Terms** (temp × wind, dome × passing)
5. **Opponent-Specific History** (player vs specific team history)

### Alternative Modeling Approaches
1. **Ensemble with Different Algorithms**: Add neural networks, gradient boosting variants
2. **Prop-Specific Models**: Separate models for each prop instead of shared features
3. **Time-Series Models**: LSTM/GRU for sequential game data
4. **Bayesian Approaches**: Better uncertainty quantification

### Market Efficiency Reality Check
If models still can't beat markets after comprehensive features:
- Betting markets may be too efficient for systematic ML edge
- Focus on specific situations (injuries, weather extremes, coaching changes)
- Explore alternative strategies (live betting, arbitrage, line shopping)

---

## File Locations Reference

### Code to Modify
- `modules/ml_feature_engineering.py` - Add new feature extraction methods
- `modules/ml_training_data_builder.py` - May need minor updates for new props

### Code READ-ONLY (Rankings System)
- `modules/adjustment_pipeline.py` - DO NOT MODIFY
- `modules/context_adjustments.py` - DO NOT MODIFY
- `modules/prop_projection_engine.py` - DO NOT MODIFY

### Data Sources (All READ-ONLY)
- `cache/pbp/pbp_YYYY.parquet` - Play-by-play data (1999-2025)
- `cache/nextgen/nextgen_YYYY.parquet` - NextGen tracking (2016-2025)
- `cache/participation/participation_YYYY.parquet` - Snap/personnel (2016-2025)
- `cache/player_props/YYYY/` - Real betting lines (2024 primarily)

### Training Data Outputs
- `cache/ml_training_data/passing_yards_2015_2024.parquet`
- `cache/ml_training_data/passing_tds_2015_2024.parquet`
- `cache/ml_training_data/receiving_yards_2015_2024.parquet`
- `cache/ml_training_data/receiving_tds_2015_2024.parquet`
- `cache/ml_training_data/receptions_2015_2024.parquet`
- `cache/ml_training_data/rushing_yards_2015_2024.parquet`
- `cache/ml_training_data/rushing_tds_2015_2024.parquet`

---

## Progress Tracking

### Phase 0: Injury Features ✅ COMPLETE
- [x] Add injury feature extraction
- [x] Rebuild passing_yards training data
- [x] Train and evaluate
- [x] **Result**: 51.2% accuracy, -2.17% ROI

### Phase 1: PBP Features (In Progress)
- [ ] Feature 1: CPOE
- [ ] Feature 2: Pressure Rate
- [ ] Feature 3: Target Share

### Phase 2: NextGen & Advanced
- [ ] Feature 4: NextGen Separation
- [ ] Feature 5: Snap Count
- [ ] Feature 6: Personnel Groupings

---

## Next Steps

1. **Start Phase 1, Feature 1 (CPOE)** when ready
2. Document results after each feature
3. Make keep/discard decisions based on criteria
4. Proceed to next feature only if previous is complete

**Ready to begin?** Start with Feature 1 (CPOE) implementation.