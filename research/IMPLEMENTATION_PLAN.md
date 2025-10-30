# GridironMetrics Enhancement Implementation Plan

## Overview

This document outlines the comprehensive plan to implement advanced contextual adjustments to NFL player performance metrics. These enhancements will layer on top of existing situational multipliers to create a sophisticated, multi-dimensional evaluation system.

---

## Current State Analysis

### Existing Infrastructure

✅ **Situational Multipliers (Already Implemented)**

- Field Position Multiplier (redzone, goalline, etc.)
- Score Differential Multiplier (game_state)
- Time Remaining Multiplier
- Down & Distance Multiplier

✅ **Data Sources**

- Play-by-play data via nflreadpy (372 columns available)
- Positional player stats (cached)
- Team stats (cached)

✅ **Architecture**

- PBP cache system (parquet files with pre-calculated multipliers)
- Polars-based data processing
- Modular calculation pipeline

---

## Phase 1: Down & Distance Context Enhancement

### Objective

Add explicit 3rd/4th down performance tracking and weighting to identify clutch performers.

### Data Available (nflreadpy)

- `down` - Down number (1-4)
- `ydstogo` - Yards to go for first down
- `third_down_converted` - Boolean for 3rd down conversions
- `fourth_down_converted` - Boolean for 4th down conversions
- `first_down_rush` / `first_down_pass` - First down gained flags

### Implementation Details

#### 1.1 Third Down Performance Multiplier

**Location:** `modules/pbp_cache_builder.py` - add to multiplier calculations

**Formula:**

```python
third_down_multiplier = pl.when(pl.col("down") == 3)
    .then(
        pl.when(pl.col("ydstogo") >= 10).then(1.5)      # 3rd & 10+: Elite difficulty
        .when(pl.col("ydstogo") >= 7).then(1.35)        # 3rd & 7-9: High difficulty
        .when(pl.col("ydstogo") >= 4).then(1.2)         # 3rd & 4-6: Medium difficulty
        .when(pl.col("ydstogo") >= 2).then(1.1)         # 3rd & 2-3: Short yardage
        .otherwise(1.05)                                 # 3rd & 1: Goalline push
    )
    .when(pl.col("down") == 4).then(1.6)  # All 4th downs = extreme clutch
    .otherwise(1.0)
```

**Rationale:**

- 3rd & long conversions are significantly more valuable than 1st/2nd down yards
- 4th down attempts show coaching confidence and clutch situations
- Progressive scaling based on yards to go

#### 1.2 Player-Level 3rd Down Stats

**Location:** New function in `main.py` - `calculate_third_down_performance()`

**Metrics to Track:**

- 3rd down conversion rate (by distance bucket)
- Average yards gained on 3rd down
- 3rd down attempts vs. conversions
- 4th down attempts vs. conversions

**Display:**
Add to player summary tables as separate columns:

- "3rd Down Conv %"
- "3rd & Long Rate" (conversions on 3rd & 7+)

#### 1.3 Integration with Overall Score

**Update:** `calculate_offensive_shares()` function

The third_down_multiplier stacks with existing situational multipliers:

```python
adjusted_value = base_stat_value 
    × field_position_multiplier 
    × game_state_multiplier 
    × time_multiplier 
    × down_multiplier  # ← ENHANCE this one
    × third_down_multiplier  # ← NEW addition
```

**Impact:**

- Player who converts 3rd & 10: gets ~1.5x credit vs same yards on 1st down
- Separates "consistent" players from "stat padders"

---

## Phase 2: Garbage Time Detection & Penalty

### Objective

Reduce credit for stats accumulated in non-competitive game situations while still giving partial credit.

### Data Available (nflreadpy)

- `score_differential` (posteam_score - defteam_score)
- `game_seconds_remaining` or `qtr` + `time`
- `posteam` - Possession team
- `defteam` - Defensive team

### Implementation Details

#### 2.1 Garbage Time Detection Logic

**Location:** `modules/pbp_cache_builder.py` - new multiplier

**Definition of Garbage Time:**

- Score differential > 17 points (3-score game)
- Time remaining < 8 minutes (late 3Q or 4Q)
- Losing team has possession

**Formula:**

```python
# Calculate time factor (0.5 to 1.0 based on time remaining)
time_factor = pl.when(pl.col("game_seconds_remaining") <= 480)  # 8 min
    .then(
        pl.max_horizontal(
            pl.lit(0.5), 
            pl.col("game_seconds_remaining") / 480
        )
    )
    .otherwise(1.0)

# Apply garbage time penalty only to losing team
garbage_time_multiplier = pl.when(
    (pl.col("score_differential").abs() > 17) & 
    (pl.col("game_seconds_remaining") <= 480) &
    (pl.col("score_differential") < 0)  # Losing team
)
.then(0.6 + (0.3 * time_factor))  # Range: 0.6 to 0.9
.otherwise(1.0)
```

**Penalty Scale:**

- Down 18-23 pts, 8 min left: ~0.9x (mild, still competitive)
- Down 24+ pts, 5 min left: ~0.72x (moderate penalty)
- Down 24+ pts, 2 min left: ~0.65x (severe penalty, pure garbage)

**Why Not Penalize Winning Team:**

- Running clock / clock management has value
- Starters often pulled, backups enter (already reflected in player stats)

#### 2.2 Interaction with Existing Multipliers

**Current game_state_multiplier already handles some of this**, but it's symmetric (penalizes both winning and losing teams equally). The new garbage_time_multiplier specifically targets:

- **Losing team putting up stats** in hopeless situations
- **More aggressive penalty** as time winds down

**Stacking:**

```python
final_value = base_stat 
    × situation_multipliers  # existing
    × garbage_time_multiplier  # NEW
```

#### 2.3 Display & Transparency

**Add to player cards:**

- "Garbage Time %" - percentage of production in garbage time situations
- Shows in tooltip: "15% of stats in garbage time (avg penalty: 0.75x)"

**Philosophy:** We don't hide garbage time stats, we just weight them appropriately.

---

## Phase 3: Personnel Grouping Inference & Context

### Objective

Weight player performance based on offensive personnel grouping (10, 11, 12 personnel) to identify players who produce in difficult formations vs. favorable matchups.

### Data Availability: ✅ **INFERABLE from nflreadpy**

While nflreadpy doesn't have an explicit `offense_personnel` column, we can **infer** personnel groupings using:

- Player positions (from cached positional data)
- Play type (pass/rush)
- Down & distance (situational tendencies)
- Receiver positions on pass plays
- Air yards (spread vs. compressed formations)
- Team tendencies (historical usage patterns)

### Personnel Grouping Definitions

- **10 Personnel:** 1 RB, 0 TE, 4 WR (spread/empty, obvious pass)
- **11 Personnel:** 1 RB, 1 TE, 3 WR (standard/base formation, ~65% of plays)
- **12 Personnel:** 1 RB, 2 TE, 2 WR (tight/compressed, balanced)
- **13 Personnel:** 1 RB, 3 TE, 1 WR (heavy, run-focused)
- **21 Personnel:** 2 RB, 1 TE, 2 WR (power run, clock management)
- **22 Personnel:** 2 RB, 2 TE, 1 WR (goal line, short yardage)

### Implementation Details

#### 3.1 Build Team Personnel Tendency Profiles

**Location:** New module `modules/personnel_analyzer.py`

**Purpose:** Analyze each team's personnel usage patterns by situation to establish baseline tendencies.

**Algorithm:**

```python
def build_team_personnel_profile(team_id: str, year: int, pbp_data: pl.DataFrame) -> dict:
    """
    Analyze team's personnel usage patterns across different situations.
    Returns tendency dictionary for inference.
    """
    
    # Group plays by situation
    situations = {
        'base': (pl.col('down') == 1) & (pl.col('ydstogo') == 10),
        'passing_downs': (pl.col('down') == 3) & (pl.col('ydstogo') >= 7),
        'short_yardage': (pl.col('ydstogo') <= 2) & (pl.col('down').is_in([3, 4])),
        'goal_line': pl.col('yardline_100') <= 5,
        'hurry_up': pl.col('no_huddle') == 1,
        'two_minute': pl.col('game_seconds_remaining') <= 120
    }
    
    # For each situation, count formation indicators:
    # - Average air yards (deep = spread, short = compressed)
    # - Receiver position distribution (TE heavy = 12/13, WR heavy = 10/11)
    # - Rush vs pass rate
    # - Success rate by formation type
    
    profile = {
        'base_personnel': '11',  # Most common formation
        'tendencies': {
            'passing_downs': {'10': 0.30, '11': 0.65, '12': 0.05},
            'short_yardage': {'22': 0.40, '13': 0.35, '21': 0.25},
            'goal_line': {'22': 0.50, '13': 0.30, '12': 0.20},
            # ... etc
        },
        'usage_rates': {
            '10': 0.12,
            '11': 0.68,
            '12': 0.15,
            '13': 0.02,
            '21': 0.02,
            '22': 0.01
        }
    }
    
    return profile
```

#### 3.2 Hybrid Personnel Inference Algorithm

**Location:** `modules/personnel_analyzer.py`

**Multi-Factor Voting System:**

```python
def infer_personnel_grouping(
    play_data: dict,
    team_profile: dict,
    player_positions: dict
) -> tuple[str, float]:
    """
    Infer offensive personnel grouping from play characteristics.
    Returns: (personnel_group, confidence_score)
    """
    
    votes = []  # List of (personnel, confidence) tuples
    
    # VOTE 1: Play Type + Receiver Position (High Confidence)
    if play_data['pass_attempt']:
        receiver_id = play_data['receiver_player_id']
        receiver_pos = player_positions.get(receiver_id, 'UNK')
        air_yards = play_data.get('air_yards', 0)
        
        if air_yards and air_yards > 20:
            # Deep shot = spread formation
            votes.append(('10', 0.75))
        elif receiver_pos == 'TE' and air_yards < 10:
            # TE short route = likely tight formation
            votes.append(('12', 0.70))
        elif receiver_pos == 'WR' and air_yards > 10:
            # WR intermediate = standard spread
            votes.append(('11', 0.80))
        elif receiver_pos == 'RB':
            # RB receiving = could be 11 or 12
            votes.append(('11', 0.60))
        else:
            votes.append(('11', 0.50))  # Default to base
    
    elif play_data['rush_attempt']:
        # Rush plays = more likely heavy formations
        if play_data['down'] >= 3 and play_data['ydstogo'] <= 2:
            # Short yardage = heavy set
            votes.append(('22', 0.70))
        elif play_data['yardline_100'] <= 5:
            # Goal line = power formation
            votes.append(('22', 0.65))
        else:
            # Standard rush = likely 11 or 21
            votes.append(('11', 0.55))
    
    # VOTE 2: Down & Distance Context (Medium Confidence)
    if play_data['down'] == 3 and play_data['ydstogo'] >= 10:
        # 3rd & long = spread to pass
        votes.append(('10', 0.60))
    elif play_data['down'] == 3 and play_data['ydstogo'] <= 2:
        # 3rd & short = heavy set
        votes.append(('22', 0.65))
    elif play_data['down'] == 1 and play_data['ydstogo'] == 10:
        # 1st & 10 = base formation
        votes.append(('11', 0.85))
    elif play_data['down'] == 2 and play_data['ydstogo'] <= 3:
        # 2nd & short = could go heavy
        votes.append(('21', 0.50))
    
    # VOTE 3: Game Script (Medium Confidence)
    score_diff = play_data.get('score_differential', 0)
    time_remaining = play_data.get('game_seconds_remaining', 3600)
    
    if abs(score_diff) > 14 and time_remaining < 600:  # 10 min
        if score_diff < 0:  # Losing team
            votes.append(('10', 0.65))  # Spread to pass and score quickly
        else:  # Winning team
            votes.append(('21', 0.60))  # Heavy to run clock
    
    # VOTE 4: Team Tendency (Lower Confidence)
    situation_key = _get_situation_key(play_data)
    if situation_key in team_profile['tendencies']:
        tendency_dist = team_profile['tendencies'][situation_key]
        # Add the most likely personnel from team tendencies
        most_likely = max(tendency_dist, key=tendency_dist.get)
        votes.append((most_likely, 0.50))
    else:
        # Fallback to team's base personnel
        votes.append((team_profile['base_personnel'], 0.40))
    
    # VOTE 5: Field Position (Low Confidence)
    if play_data['yardline_100'] <= 3:
        votes.append(('22', 0.55))  # Goal line = jumbo
    elif play_data['yardline_100'] >= 80:
        votes.append(('11', 0.50))  # Own territory = conservative base
    
    # AGGREGATE VOTES (weighted by confidence)
    personnel_scores = {}
    for personnel, confidence in votes:
        if personnel not in personnel_scores:
            personnel_scores[personnel] = 0.0
        personnel_scores[personnel] += confidence
    
    # Select highest scoring personnel
    best_personnel = max(personnel_scores, key=personnel_scores.get)
    
    # Calculate aggregate confidence (average of votes for winner)
    winner_votes = [conf for pers, conf in votes if pers == best_personnel]
    avg_confidence = sum(winner_votes) / len(votes) if votes else 0.5
    
    # Normalize confidence to 0.0-1.0 range
    normalized_confidence = min(1.0, avg_confidence)
    
    return best_personnel, normalized_confidence


def _get_situation_key(play_data: dict) -> str:
    """Helper to categorize play situation for tendency lookup."""
    down = play_data['down']
    ydstogo = play_data['ydstogo']
    yardline = play_data.get('yardline_100', 50)
    
    if yardline <= 5:
        return 'goal_line'
    elif ydstogo <= 2 and down >= 3:
        return 'short_yardage'
    elif down == 3 and ydstogo >= 7:
        return 'passing_downs'
    elif down == 1 and ydstogo == 10:
        return 'base'
    else:
        return 'other'
```

#### 3.3 Personnel Grouping Multipliers

**Location:** `modules/pbp_cache_builder.py` - add to multiplier calculations

**Position-Specific Multipliers:**

```python
# For Wide Receivers
wr_personnel_multiplier = pl.when(pl.col("inferred_personnel") == '10')
    .then(1.15)   # 4 WR set = defense spread thin, easier to get open
    .when(pl.col("inferred_personnel") == '11')
    .then(1.0)    # Standard formation = neutral
    .when(pl.col("inferred_personnel") == '12')
    .then(0.90)   # 2 TE set = defense compressed, harder separation
    .when(pl.col("inferred_personnel").is_in(['13', '22']))
    .then(0.85)   # Heavy sets = obvious run, WR not focal point
    .otherwise(1.0)

# For Running Backs
rb_personnel_multiplier = pl.when(pl.col("inferred_personnel") == '10')
    .then(0.90)   # Spread = RB not focal, likely passing down
    .when(pl.col("inferred_personnel") == '11')
    .then(1.0)    # Standard = neutral
    .when(pl.col("inferred_personnel") == '12')
    .then(1.05)   # Tight set = more blocking, easier running lanes
    .when(pl.col("inferred_personnel").is_in(['21', '22']))
    .then(1.10)   # Heavy = obvious run, defense loads box BUT proves skill
    .when(pl.col("inferred_personnel") == '13')
    .then(1.08)   # 3 TE = compressed, good for power running
    .otherwise(1.0)

# For Tight Ends
te_personnel_multiplier = pl.when(pl.col("inferred_personnel") == '10')
    .then(0.85)   # No TE or 1 TE in 4 WR set = not involved much
    .when(pl.col("inferred_personnel") == '11')
    .then(1.0)    # Standard 1 TE = neutral
    .when(pl.col("inferred_personnel").is_in(['12', '13']))
    .then(1.10)   # Multiple TE sets = TE-focused offense, more opportunities
    .when(pl.col("inferred_personnel").is_in(['21', '22']))
    .then(0.95)   # Heavy RB sets = TE more blocking than receiving
    .otherwise(1.0)

# Apply multiplier only when confidence > 0.6
final_personnel_multiplier = pl.when(pl.col("personnel_confidence") > 0.6)
    .then(pl.col("position_specific_personnel_multiplier"))
    .otherwise(1.0)  # Don't apply if uncertain
```

**Rationale:**

- **10 Personnel** (4 WR): WRs benefit (defense spread), RBs penalized (likely pass)
- **11 Personnel** (3 WR): Neutral baseline, most common formation
- **12 Personnel** (2 TE): TEs benefit (more involved), WRs penalized (compressed)
- **Heavy Sets** (21, 22, 13): RBs get boost if they succeed (defense loaded box), WRs penalized (not focal point)

#### 3.4 Confidence Thresholds

**Only apply personnel multiplier when confidence ≥ 60%**

**Expected Confidence Distribution:**

- High Confidence (>80%): ~30% of plays
  - 1st & 10 standard downs
  - Goal line situations
  - 3rd & long obvious passing downs
  
- Medium Confidence (60-80%): ~45% of plays
  - Standard 2nd downs
  - Pass plays with clear receiver type
  - Short yardage situations
  
- Low Confidence (<60%): ~25% of plays
  - Unusual down/distance
  - Trick plays
  - Ambiguous situations
  
For low confidence plays, default to **1.0x multiplier** (no adjustment).

#### 3.5 Validation & Calibration

**Testing Strategy:**

1. **Spot Check Known Formations:**
   - Goal line plays → Should predict 22/13 with high confidence
   - 3rd & 15 → Should predict 10/11 with high confidence
   - 1st & 10 → Should predict 11 with very high confidence

2. **Team Profile Validation:**
   - Teams known for spread offense (KC, MIA) → High % of 10/11 personnel
   - Teams known for heavy formations (BAL, SF) → Higher % of 12/21/22

3. **Position Success Rates:**
   - WRs should perform better in 10 personnel (validate with actual stats)
   - RBs should show success in heavy sets even with stacked boxes

4. **Confidence Calibration:**
   - Track prediction accuracy by confidence bucket
   - If <60% confidence plays are wrong >50% of time, threshold is working
   - If >80% confidence plays are wrong >20% of time, algorithm needs tuning

#### 3.6 Display in Output

**Add columns to player cards:**

```txt
Player: Tyreek Hill (WR)
Personnel Usage:
  - 10 Personnel: 18% of snaps (Avg: 1.15x multiplier)
  - 11 Personnel: 72% of snaps (Avg: 1.0x multiplier)  
  - 12 Personnel: 10% of snaps (Avg: 0.90x multiplier)
  
Versatility Score: 0.92 (performs well across formations)
```

**Versatility Score Formula:**

```python
versatility = (
    production_in_10_personnel / expected_10_production +
    production_in_11_personnel / expected_11_production +
    production_in_12_personnel / expected_12_production
) / 3

# >1.0 = Versatile (produces in all formations)
# ~1.0 = Neutral (formation-dependent)
# <1.0 = One-dimensional (only produces in favorable formations)
```

#### 3.7 Integration with Overall Score

Personnel multiplier stacks with existing multipliers:

```python
final_play_value = base_stat_value 
    × field_position_multiplier 
    × game_state_multiplier 
    × time_multiplier 
    × third_down_multiplier
    × garbage_time_multiplier
    × personnel_multiplier  # ← NEW (position-specific)
```

**Impact Examples:**

- WR catching 40-yard TD in 10 personnel (4 WR spread): Base × 1.15 = +15% boost
- Same TD in 22 personnel (heavy set): Base × 0.85 = -15% penalty (impressive but unusual)
- RB rushing for 80 yards in 22 personnel (stacked box): Base × 1.10 = +10% boost (earned it vs loaded box)

#### 3.8 Limitations & Transparency

**Document Known Limitations:**

1. **Inference Accuracy:** ~70-80% accurate, not 100% perfect
2. **Confidence Required:** Only applied when confidence ≥ 60%, otherwise 1.0x
3. **Position Assumptions:** Assumes standard position usage (could be wrong on trick plays)
4. **No Exotic Formations:** Doesn't handle wildcat, empty backfield with 5 WR, etc.

**Show in reports:**

```txt
Personnel Inference Accuracy: 76% (validated on 2024 season)
High Confidence Plays: 73% of dataset
Personnel Multiplier Applied: 73% of plays
```

---

## Phase 4: Blocking Quality Proxy (Yards Before/After Contact)

### Objective

Estimate blocking quality to differentiate players creating their own yards vs. benefiting from great blocking.

### Data Available (nflreadpy)

✅ **For Receivers:**

- `air_yards` - Yards ball traveled in air
- `yards_after_catch` - YAC
- `yards_gained` - Total yards

✅ **For Rushers:**

- `yards_gained` - Total rushing yards
- ❌ **yards_before_contact** - NOT in nflreadpy
- ❌ **yards_after_contact** - NOT in nflreadpy

### Implementation Details

#### 4.1 Receiver YAC Multiplier

**Available & Implementable**

**Formula:**

```python
# Calculate YAC percentage of total yards
yac_percentage = pl.col("yards_after_catch") / pl.col("yards_gained")

# Reward high YAC (creating own yards)
yac_multiplier = pl.when(yac_percentage > 0.7).then(1.15)  # 70%+ YAC
    .when(yac_percentage > 0.5).then(1.1)   # 50-70% YAC
    .when(yac_percentage > 0.3).then(1.05)  # 30-50% YAC
    .when(yac_percentage < 0.1).then(0.95)  # <10% YAC (contested/jump balls only)
    .otherwise(1.0)
```

**Rationale:**

- High YAC % = receiver breaking tackles, eluding defenders
- Low YAC % = either contested catches (good) or QB perfectly placed (QB credit)
- Separates "yards creators" from "get open and catch" receivers

#### 4.2 Rush Yards Before Contact - UNAVAILABLE

**Problem:** nflreadpy doesn't provide yards before/after contact for rushers

**Workaround Options:**

**Option A:** Use Average Yards Per Carry as Proxy

- RBs with high YPC on high volume likely have good blocking
- Compare player YPC to team average RB YPC
- If player >> team average, probably creating own yards

**Option B:** Compare Rushing EPA to Expected

- If player's rush_epa >> expected based on down/distance/field position
- Indicates player is exceeding blocking quality

**Option C:** Skip Until Better Data Available
**Recommendation:** Use **Option A** as a rough proxy

**Formula:**

```python
# In player analysis, calculate:
player_ypc = total_rush_yards / total_rush_attempts
team_avg_ypc = (team_total_rush_yards - player_rush_yards) / (team_attempts - player_attempts)

blocking_context = player_ypc / team_avg_ypc
# If 1.2+: Player likely creating own yards (boost)
# If 0.8-: Player likely dependent on blocking (neutral/slight penalty)
```

#### 4.3 Integration

**Location:** Post-processing adjustment in `generate_top_contributors()`

Apply as a **small adjustment** (5-10% max) since it's imperfect proxy data:

```python
blocking_adjustment = pl.when(blocking_context > 1.2).then(1.05)
    .when(blocking_context < 0.8).then(0.97)
    .otherwise(1.0)
```

---

## Phase 5: Catch Rate vs. League Average

### Objective

Identify receivers who consistently catch difficult targets vs. those benefiting from perfect QB placement.

### Data Available (nflreadpy)

✅ `complete_pass` - Binary flag for completion
✅ `incomplete_pass` - Binary flag for incompletion  
✅ `receiver_player_id` - Target receiver
✅ `air_yards` - Difficulty proxy (deep targets harder)

### Implementation Details

#### 5.1 Calculate Player Catch Rate

**Location:** New function in `main.py`

```python
def calculate_catch_rates(player_stats: pl.DataFrame, pbp_data: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate catch rate for each receiver and compare to league average.
    """
    # Get all targets (completions + incompletions) from PBP
    targets = pbp_data.filter(
        (pl.col("complete_pass") == 1) | (pl.col("incomplete_pass") == 1)
    ).group_by(["receiver_player_id", "receiver_player_name"]).agg([
        pl.col("complete_pass").sum().alias("receptions"),
        pl.len().alias("targets"),
        pl.col("air_yards").mean().alias("avg_target_depth")
    ])
    
    # Calculate catch rate
    targets = targets.with_columns([
        (pl.col("receptions") / pl.col("targets")).alias("catch_rate")
    ])
    
    # Calculate league average (weighted by targets)
    league_avg_catch_rate = (
        targets["receptions"].sum() / targets["targets"].sum()
    )
    
    # Calculate catch rate over expected (CRoE)
    targets = targets.with_columns([
        (pl.col("catch_rate") - league_avg_catch_rate).alias("catch_rate_over_expected")
    ])
    
    return targets
```

#### 5.2 Adjust Player Scores

**Apply in:** `generate_top_contributors()`

**Formula:**

```python
# Boost receivers with high CRoE (creating separation, making tough catches)
catch_rate_multiplier = pl.when(pl.col("catch_rate_over_expected") > 0.05).then(1.08)
    .when(pl.col("catch_rate_over_expected") > 0.02).then(1.04)
    .when(pl.col("catch_rate_over_expected") < -0.05).then(0.96)
    .when(pl.col("catch_rate_over_expected") < -0.02).then(0.98)
    .otherwise(1.0)
```

**Rationale:**

- +5% catch rate over average = excellent hands/separation (1.08x boost)
- -5% catch rate = likely QB issues or running wrong routes (0.96x penalty)

#### 5.3 Control for Target Depth

Important: Deep targets (air_yards > 20) have lower expected catch rates.

**Enhanced Formula:**

```python
# Adjust expected catch rate based on target depth
expected_catch_rate = pl.when(pl.col("avg_target_depth") > 20).then(0.45)  # Deep
    .when(pl.col("avg_target_depth") > 15).then(0.55)  # Intermediate
    .when(pl.col("avg_target_depth") > 10).then(0.65)  # Medium
    .otherwise(0.75)  # Short

catch_rate_over_expected = pl.col("catch_rate") - expected_catch_rate
```

#### 5.4 Display

Add to WR/TE player cards:

- "Catch Rate: 68.5% (League: 65.2%, +3.3%)"
- Color code: Green if above average, red if below

---

## Phase 6: Skill Position Talent Adjustment

### Objective

Apply retroactive adjustment based on quality of surrounding skill position talent. Boost players carrying weak offenses, slight penalty for players surrounded by elite talent.

### Implementation Details

#### 6.1 Calculate Baseline Scores (First Pass)

**No changes to existing calculation pipeline**

Run the full analysis through Phase 1-5 to get:

- Player overall_contribution scores
- Position rankings
- All situational adjustments applied

**Result:** Unadjusted baseline scores for all players

#### 6.2 Calculate Teammate Quality Index

**Location:** New function `calculate_teammate_quality()`

**For Each Player, Define "Teammates":**

- **RB:** All other RBs on same team + WR1 + TE1 + QB
- **WR:** All other WRs on same team + RB1 + TE1 + QB  
- **TE:** All WRs on same team + RB1 + other TEs + QB

**Note:** QB excluded from adjustment initially (as discussed)

**Formula:**

```python
def calculate_teammate_quality(
    player_scores: pl.DataFrame,
    player_stats: pl.DataFrame
) -> pl.DataFrame:
    """
    Calculate the aggregate quality of each player's skill position teammates.
    """
    # For each player, sum the scores of their teammates
    teammate_scores = []
    
    for player_row in player_scores.iter_rows(named=True):
        player_id = player_row["player_id"]
        player_name = player_row["player_name"]
        team = player_row["team"]
        position = player_row["position"]
        player_score = player_row["adjusted_score"]
        
        # Get all skill position players on same team
        teammates = player_scores.filter(
            (pl.col("team") == team) &
            (pl.col("position").is_in(["RB", "WR", "TE"])) &
            (pl.col("player_id") != player_id)  # Exclude self
        )
        
        # Sum teammate scores
        teammate_total = teammates["adjusted_score"].sum()
        teammate_count = len(teammates)
        teammate_avg = teammate_total / teammate_count if teammate_count > 0 else 0
        
        teammate_scores.append({
            "player_id": player_id,
            "player_name": player_name,
            "teammate_total_score": teammate_total,
            "teammate_avg_score": teammate_avg,
            "teammate_count": teammate_count
        })
    
    return pl.DataFrame(teammate_scores)
```

#### 6.3 Convert to Percentile Rankings

**Formula:**

```python
# Convert teammate_total_score to percentile (0-100)
teammate_quality_pct = teammate_scores.with_columns([
    pl.col("teammate_total_score").rank(descending=True) 
        / pl.col("teammate_total_score").count() * 100
]).alias("teammate_quality_percentile")

# Low percentile (0-25%) = Weak supporting cast → Boost player
# High percentile (75-100%) = Strong supporting cast → Slight penalty
```

#### 6.4 Apply Talent Adjustment Multiplier

**Location:** Update `generate_top_contributors()` after baseline calculation

**Formula:**

```python
talent_adjustment = pl.when(pl.col("teammate_quality_percentile") < 25)
    .then(1.10)  # Top 25% hardest context (weak teammates) → 10% boost
    .when(pl.col("teammate_quality_percentile") < 40)
    .then(1.05)  # 25-40% → 5% boost
    .when(pl.col("teammate_quality_percentile") > 75)
    .then(0.95)  # Bottom 25% easiest context (elite teammates) → 5% penalty
    .when(pl.col("teammate_quality_percentile") > 60)
    .then(0.97)  # 60-75% → 3% penalty
    .otherwise(1.0)  # 40-60%: No adjustment (average context)

# Apply adjustment
final_score = adjusted_score × talent_adjustment
```

**Rationale:**

- Adjustments are conservative (±5-10% max)
- Avoids over-correcting and creating nonsensical rankings
- Recognizes both "carrying the offense" and "benefiting from spacing"

#### 6.5 Sample Size Dampening with 0.4 Root Curve

**Apply to final_score to reduce noise from small samples**

**Formula:**

```python
# Calculate sample size factor
games_played_factor = (pl.col("games_played") ** 0.4) / (17 ** 0.4)

# Apply to final score
sample_adjusted_score = final_score × games_played_factor
```

**Effect:**

- 17 games: 1.0x (full credit)
- 10 games: 0.75x credit
- 5 games: 0.57x credit  
- 2 games: 0.39x credit

**Why 0.4 root:**

- Balances between:
  - Not over-penalizing injury-shortened seasons
  - Properly dampening hot streaks from limited action
- More aggressive than sqrt (0.5) but less than cubic root (0.33)

#### 6.6 Display Talent Context

**Add to player cards:**

- "Supporting Cast: 23rd percentile (Weak) → +10% boost"
- "Supporting Cast: 87th percentile (Elite) → -5% penalty"

**Color coding:**

- Red (0-25%): Hard carry situation
- Yellow (25-75%): Average context
- Green (75-100%): Surrounded by talent

---

## Phase 7: Integration & Calculation Pipeline

### Complete Calculation Flow

```txt
┌─────────────────────────────────────────────────────────────────┐
│ 1. Load Raw Data                                                │
│    - Player stats (from cache)                                  │
│    - Team stats (from cache)                                    │
│    - Play-by-play data (from cache with situational multipliers)│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Apply Play-Level Multipliers (PBP Cache)                     │
│    ✓ Field position multiplier                                  │
│    ✓ Game state multiplier (score diff)                         │
│    ✓ Time remaining multiplier                                  │
│    ✓ Down & distance multiplier                                 │
│    → Third down clutch multiplier                               │
│    → Garbage time multiplier                                    │
│    → YAC multiplier (receivers only)                            │
│    → Personnel grouping multiplier (position-specific)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Aggregate to Player Level                                    │
│    - Sum adjusted stats by player                               │
│    - Calculate offensive share percentages                      │
│    - Apply position-specific weightings                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Calculate Baseline Player Scores (First Pass)                │
│    - Overall contribution score (unadjusted)                    │
│    - Positional rankings                                        │
│    - Store as "baseline_score"                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Calculate Context Metrics                                    │
│    → Catch rate over expected (WR/TE)                           │
│    → Blocking quality proxy (RB: YPC vs team avg)               │
│    → Teammate quality index (all skill positions)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Apply Retroactive Adjustments (Second Pass)                  │
│    → Catch rate adjustment (±4-8%)                              │
│    → Blocking quality adjustment (±3-5%)                        │
│    → Talent context adjustment (±5-10%)                         │
│    = context_adjusted_score                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. Apply Sample Size Dampening                                  │
│    → (games_played ^ 0.4) / (17 ^ 0.4)                          │
│    = final_score                                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. Generate Rankings & Output                                   │
│    - Top contributors table                                     │
│    - Positional rankings (RB, WR, TE)                           │
│    - QB separation (existing)                                   │
│    - Context transparency columns                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 8: Code Architecture Changes

### 8.1 New Files to Create

#### `modules/context_adjustments.py`

```python
"""
Context-aware adjustments for player performance.
Includes catch rate, blocking quality proxy, and teammate quality calculations.
"""

def calculate_catch_rate_adjustment(player_stats, pbp_data) -> pl.DataFrame:
    """Calculate catch rate over expected for receivers."""
    pass

def calculate_blocking_quality_proxy(player_stats, team_stats) -> pl.DataFrame:
    """Estimate blocking quality from YPC differential."""
    pass

def calculate_teammate_quality_index(player_scores) -> pl.DataFrame:
    """Calculate aggregate quality of skill position teammates."""
    pass

def apply_talent_adjustment(player_scores, teammate_quality) -> pl.DataFrame:
    """Apply retroactive talent context adjustment."""
    pass

def apply_sample_size_dampening(player_scores, root=0.4) -> pl.DataFrame:
    """Apply games_played ^ root dampening curve."""
    pass
```

### 8.2 Modified Files

#### `modules/pbp_cache_builder.py`

**Add new multipliers:**

- `third_down_multiplier` (based on down + ydstogo)
- `garbage_time_multiplier` (based on score_diff + time + possession)
- `yac_multiplier` (for pass plays with YAC data)

**Update `_calculate_multipliers()` function**

#### `main.py`

**Modify `calculate_offensive_shares()`:**

- Load new multipliers from PBP cache
- Apply in calculation pipeline

**Modify `generate_top_contributors()`:**

1. Calculate baseline scores (existing logic)
2. Call `calculate_catch_rate_adjustment()`
3. Call `calculate_blocking_quality_proxy()`
4. Call `calculate_teammate_quality_index()`
5. Apply all context adjustments
6. Apply sample size dampening
7. Generate final rankings

**Add new columns to output tables:**

- "Context Adj" - Show talent adjustment applied
- "Catch Rate" - Show for WR/TE
- "3rd Down %" - Show conversion rate
- "Garbage Time %" - % of production in garbage time

### 8.3 Cache Rebuild Required

After implementing new multipliers in `pbp_cache_builder.py`:

```bash
python -m modules.pbp_cache_builder --start-year 2000 --end-year 2025 --force
```

**Estimated time:** ~2.5 hours (26 years × ~6 seconds per year)

---

## Phase 9: Testing & Validation Strategy

### 9.1 Unit Tests

Create `tests/test_context_adjustments.py`:

```python
def test_third_down_multiplier():
    """Verify 3rd & 10 gets 1.5x, 3rd & 1 gets 1.05x"""
    pass

def test_garbage_time_detection():
    """Verify down 24 with 2 min = ~0.65x penalty"""
    pass

def test_talent_adjustment_range():
    """Verify adjustments stay within 0.85-1.15 range"""
    pass

def test_sample_size_dampening():
    """Verify 0.4 root curve calculations"""
    pass
```

### 9.2 Integration Tests

**Test with 2024 season data:**

1. Run full analysis with new features
2. Verify top 10 RB/WR/TE rankings are reasonable
3. Check for:
   - Negative scores (shouldn't happen)
   - Extreme outliers (investigate if found)
   - Garbage time % makes sense (check known garbage time games)

### 9.3 Validation Against Known Cases

**Test Case 1: Garbage Time Specialist**

- Find a player known for garbage time stats (e.g., certain QBs/WRs on bad teams)
- Verify their garbage_time_percentage is high (>20%)
- Verify score is appropriately reduced

**Test Case 2: Elite Receiver in Crowded Room**

- Test with 49ers WRs (Deebo, Aiyuk, Kittle all elite)
- Verify talent adjustment applied (slight penalty)
- Verify adjustment is small (3-5%)

**Test Case 3: Bellcow RB on Bad Team**

- Test with Derrick Henry (high usage, weak supporting cast)
- Verify talent adjustment boost applied
- Verify 3rd down performance properly weighted

### 9.4 Comparative Analysis

**Before/After Rankings:**

1. Run analysis with old system (current)
2. Run analysis with new system (all features)
3. Document top 20 movers in each direction
4. Investigate why they moved (which adjustments drove change)
5. Validate changes make logical sense

---

## Phase 10: Documentation & Transparency

### 10.1 Update README.md

Document all new features:

- What they measure
- Why they matter
- How they're calculated
- Adjustment ranges

### 10.2 Methodology Document

Create `METHODOLOGY.md` with:

- Complete formula documentation
- Rationale for each multiplier
- Adjustment ranges and reasoning
- Data sources and limitations

### 10.3 Output Transparency

**In generated reports, show:**

- Base score vs. adjusted score
- Breakdown of adjustments applied:

```txt
  Player: Christian McCaffrey
  Base Score: 450.2
  + Situational Adj: +112.5 (25%)
  - Talent Context: -22.5 (5% penalty - elite supporting cast)
  + Sample Size: Full season (no dampening)
  = Final Score: 540.2
```

### 10.4 Interactive Tooltips (Future)

If building a web frontend:

- Hover over scores to see adjustment breakdown
- Click player to see detailed context metrics
- Filter by "garbage time removed" or "with garbage time"

---

## Phase 11: Implementation Timeline

### Week 1: Foundation

- ✅ Review current codebase (DONE)
- ✅ Verify data availability (DONE)
- [ ] Create `modules/context_adjustments.py`
- [ ] Set up test framework

### Week 2: PBP Multipliers (Base Layer)

- [ ] Implement `third_down_multiplier` in pbp_cache_builder
- [ ] Implement `garbage_time_multiplier` in pbp_cache_builder
- [ ] Implement `yac_multiplier` in pbp_cache_builder
- [ ] Rebuild PBP cache for all years
- [ ] Test multipliers on sample data

### Week 3: Personnel Grouping System

- [ ] Create `modules/personnel_analyzer.py`
- [ ] Implement `build_team_personnel_profile()`
- [ ] Implement `infer_personnel_grouping()` with multi-factor voting
- [ ] Add position-specific personnel multipliers to pbp_cache_builder
- [ ] Test inference accuracy on 2024 data
- [ ] Rebuild PBP cache with personnel inference

### Week 4: Player-Level Context

- [ ] Implement `calculate_catch_rate_adjustment()`
- [ ] Implement `calculate_blocking_quality_proxy()`
- [ ] Add to calculation pipeline in main.py
- [ ] Test with 2024 season

### Week 5: Talent Adjustment

- [ ] Implement `calculate_teammate_quality_index()`
- [ ] Implement `apply_talent_adjustment()`
- [ ] Implement `apply_sample_size_dampening()`
- [ ] Full integration test

### Week 6: Validation & Refinement

- [ ] Run comparative analysis (old vs new)
- [ ] Validate adjustment ranges
- [ ] Validate personnel inference accuracy
- [ ] Tune multiplier values if needed
- [ ] Document edge cases

### Week 7: Documentation & Polish

- [ ] Update README.md
- [ ] Create METHODOLOGY.md
- [ ] Add transparency columns to output
- [ ] Final testing across multiple seasons

---

## Phase 12: Future Enhancements (Post-V1)

### Advanced Blocking Metrics (if data becomes available)

- Yards before contact for RBs
- Pressure rate for QBs
- Time to throw
- Separation at catch for WRs

### Machine Learning Enhancements

- Train model to predict "expected points added" given context
- Use actual vs. expected as adjustment factor
- Requires significant historical data

### QB Integration for Talent Adjustment

- Inverse relationship (elite QB = easier for WRs)
- Chicken-egg iteration to handle feedback loop
- May need multiple passes to converge

---

## Summary: What We're Building

### Inputs

1. ✅ Play-by-play data (nflreadpy)
2. ✅ Player stats (cached)
3. ✅ Team stats (cached)

### Outputs

1. **Enhanced Player Scores** with multi-layer adjustments
2. **Contextual Metrics** (catch rate, teammate quality, garbage time %)
3. **Transparent Breakdowns** showing how adjustments were applied

### Key Differentiators

- **Situational Context:** Not just what they did, but when and how
- **Clutch Performance:** 3rd down conversions weighted appropriately
- **Garbage Time Filtering:** Separate meaningful production from stat padding
- **Talent Context:** Recognize players elevating weak offenses
- **Sample Size Aware:** Proper dampening for injury/limited action
- **Transparent Methodology:** Users can see exactly why rankings are what they are

### What Makes This Special

This isn't just another stat aggregator. This is a **sophisticated, multi-dimensional evaluation system** that:

- Captures game context (field position, score, time)
- Captures play context (3rd down, garbage time, YAC creation)
- Captures team context (supporting cast quality)
- Captures sample size confidence
- Presents it all transparently

**Result:** The most contextually-aware NFL player evaluation system available for free.

---

## Risk Mitigation

### Risk 1: Over-adjustment

**Mitigation:** Keep all adjustments conservative (5-15% max per factor)

### Risk 2: Data Quality Issues

**Mitigation:** Extensive validation, handle nulls gracefully, document limitations

### Risk 3: Interpretation Complexity

**Mitigation:** Clear documentation, tooltips, transparency in output

### Risk 4: Performance/Speed

**Mitigation:** Use cached PBP data, polars for speed, parallel processing where possible

### Risk 5: Controversial Rankings

**Mitigation:** Document methodology thoroughly, show before/after comparisons, be transparent about limitations

---

## Success Metrics

### Quantitative

- [ ] All seasons 2000-2025 processed successfully
- [ ] Processing time < 5 minutes per season
- [ ] Zero negative scores
- [ ] Adjustment ranges within expected bounds

### Qualitative

- [ ] Top 10 rankings pass "smell test" (recognizable elite players)
- [ ] Garbage time players properly identified
- [ ] 3rd down specialists elevated appropriately
- [ ] Supporting cast context makes logical sense

### Community

- [ ] Clear documentation that non-technical users can understand
- [ ] Transparent methodology that advanced users can critique
- [ ] Unique insights not available elsewhere

---

## Conclusion

This implementation plan provides a **complete, production-ready roadmap** to transform GridironMetrics from a solid stats aggregator into a **sophisticated player evaluation system** that accounts for:

- Situational leverage
- Clutch performance
- Garbage time inflation
- Supporting cast quality
- Sample size confidence

The approach is **data-driven, transparent, and conservative** in its adjustments, ensuring results are both sophisticated and defensible.

**Estimated Total Implementation Time:** 6 weeks part-time or 2-3 weeks full-time

**Data Requirements:** All available through existing nflreadpy integration ✅

**Cost:** $0 (all free/open source) ✅

**Differentiation:** High - no other free tool does this comprehensively ✅
