# GridironMetrics

**Context-Aware NFL Offensive Player Evaluation System**

## Purpose

GridironMetrics goes beyond traditional volume-based statistics to evaluate the **true offensive impact** of NFL skill position players and quarterbacks. Rather than simply counting yards and touchdowns, this system measures **quality-adjusted contribution per opportunity**, accounting for the difficulty of each situation a player faces.

The goal is to answer: *"How much did this player contribute to their team's offensive success, considering the context of every play they participated in?"*

By weighting each play based on 10+ contextual factors—including field position, score differential, down-and-distance, personnel groupings, defensive alignments, and coverage schemes—GridironMetrics reveals which players consistently deliver high-impact performances in difficult situations, not just those who accumulate stats in favorable circumstances.

---

## Position-Specific Features

### Running Backs (RB)

**Core Metrics:**
- Rushing yards, rushing touchdowns, carries
- Receiving yards, receiving touchdowns, receptions, targets

**Contextual Adjustments:**
- **Defenders in Box**: Adjusts for facing stacked boxes (8+ defenders)
  - 8+ defenders: 1.25x multiplier
  - 7 defenders: 1.15x
  - 6 defenders: 1.05x
  - 4-5 defenders: 0.90-0.95x
- **Personnel Groupings**: Accounts for offensive formation difficulty
  - Heavy formations (21/22): Easier rushing situations (0.85x)
  - 11 personnel vs 8+ box: Tougher rushing (1.10x)
- **Blocking Quality**: Adjusts for yards before contact and tackle breaks
- **YAC Efficiency**: Rewards explosive plays after initial contact
- **Catch Rate**: Adjusts receiving performance vs expected completion percentage

**FTN Adjustments (2022+):**
- **RPO Reduction (-12%)**: RPO runs have numbers advantage (QB reads to favorable side)
- **Heavy Box Bonus (+15%)**: Running against 8+ defenders is significantly harder (verified count)

**Output Includes:**
- Average defenders in box faced
- Percentage of rushes vs 8+ defenders
- Average difficulty multiplier (1.020-1.050 typical range)
- Consistency metrics (floor/ceiling/typical game)

---

### Wide Receivers (WR)

**Core Metrics:**
- Receiving yards, receiving touchdowns, receptions, targets
- Air yards, yards after catch

**Contextual Adjustments:**
- **Coverage Type**: Adjusts for facing man vs zone coverage
  - Man coverage (Cover 0, Cover 1, 2-Man): 1.15x multiplier
  - Zone coverage (Cover 2, Cover 3, Cover 4): 1.00x
  - Prevent defense: 0.90x
- **Personnel Groupings**: Accounts for route-running difficulty
  - Spread formations (00/10): Easier separation (0.90x in prevent)
  - Heavy formations with WRs: Tougher assignments (1.05x)
- **Target Quality**: Adjusts for catchable ball rate and contested catches
- **Catch Rate**: Performance vs expected completion percentage
- **YAC Efficiency**: Rewards explosive gains after the catch
- **Separation**: Implicitly measured through coverage type and catch difficulty

**FTN Adjustments (2022+):**
- **Contested Catch Bonus (+25%)**: Catching with defender in position requires elite ball skills
- **Drop Penalty (-8 pts per drop)**: Drops represent unreliability and missed opportunities
- **Screen Pass Reduction (-10%)**: Screen catches are easier (less traffic, designed space)

**Output Includes:**
- Percentage of targets vs man coverage
- Average coverage difficulty multiplier (1.015-1.040 typical range)
- Target volume and efficiency metrics
- Consistency across games

---

### Tight Ends (TE)

**Core Metrics:**
- Receiving yards, receiving touchdowns, receptions, targets
- Blocking snaps and effectiveness

**Contextual Adjustments:**
- **Coverage Type**: Same as WR adjustments for receiving routes
  - Man coverage: 1.15x
  - Zone coverage: 1.00x
- **Personnel Groupings**: Dual-role difficulty
  - Heavy formations (12/13/22): Blocking-heavy situations (0.85x receiving)
  - 11 personnel: Route-running emphasis (1.00x)
- **Blocking Quality**: Adjusts contribution in run-blocking situations
  - Pass protection effectiveness
  - Run-blocking efficiency
- **Catch Rate**: Adjusts for contested catches and traffic receptions
- **YAC Efficiency**: Middle-of-field yards after catch

**FTN Adjustments (2022+):**
- **Contested Catch Bonus (+25%)**: Catching with defender in position requires elite ball skills
- **Drop Penalty (-8 pts per drop)**: Drops represent unreliability and missed opportunities
- **Screen Pass Reduction (-10%)**: Screen catches are easier (less traffic, designed space)

**Output Includes:**
- Coverage type distribution
- Blocking vs receiving balance
- Average difficulty multiplier (1.010-1.030 typical range)
- Dual-threat value assessment

---

### Quarterbacks (QB)

**Core Metrics:**
- Passing yards, passing touchdowns, completions, attempts, interceptions
- Rushing yards, rushing touchdowns
- Completion percentage, yards per attempt

**Contextual Adjustments:**
- **Down & Distance**: High-leverage situations weighted more heavily
  - 3rd/4th down conversions: 1.2-1.5x multiplier based on distance
  - 1st/2nd down: 1.0x baseline
- **Field Position**: Scoring opportunities weighted by field location
  - Red zone (inside 20): 1.3x
  - Plus territory (opp 21-50): 1.15x
  - Own territory: 0.85-1.0x
- **Score Differential**: Clutch performance emphasis
  - Trailing by 1-8: 1.2x
  - Close games (±3): 1.15x
  - Leading comfortably: 0.9x
- **Time Remaining**: Late-game importance
  - Final 2 minutes of half: 1.15x
  - 4th quarter: 1.05-1.1x
- **Pass Protection**: Implicitly measured through sack avoidance
- **Garbage Time**: Prevents stat-padding (0.5x multiplier when game decided)

**Advanced Adjustments (2024+):**
- **Pressure Bonuses**: Completions under pressure (NextGen Stats GPS tracking)
  - Completion under pressure: +40% points
  - Yards under pressure: +20% value
- **Contextual Penalties**: Turnovers/sacks weighted by game situation
  - Field position multiplier: 1.5x in own territory
  - Score differential: 1.2x in close games
  - Time remaining: 1.3x in final 2 minutes
  - Down/distance: 0.7x on 3rd/4th down (calculated risk)

**FTN Adjustments (2022+):**
- **Play Action Reduction (-10%)**: Play action passes are easier (defense fooled, simplified reads)
- **Out of Pocket Bonus (+3 pts per completion)**: Completing while scrambling shows improvisation
- **Blitz Adjustment (-2 pts per completion)**: Blitz creates easier throwing windows (less coverage)
- **Screen Pass Reduction (-15%)**: Screens are high-percentage, low-difficulty throws

**Minimum Activity Threshold:** 14 pass attempts per game

**Output Includes:**
- Efficiency metrics (yards/attempt, TD%, INT%)
- Clutch performance indicators
- Consistency and trending
- High-leverage play success rate

---

## Universal Contextual Multipliers

These adjustments apply to **all positions**:

### Situational Context
- **Field Position** (0.85x - 1.3x): Value of plays based on field location
- **Score Differential** (0.9x - 1.2x): Emphasizes competitive situations
- **Time Remaining** (0.95x - 1.15x): Weights late-game importance
- **Down & Distance** (0.9x - 1.5x): Rewards converting difficult downs
- **Garbage Time** (0.5x): Heavily discounts stat-padding when game is decided

### Performance Quality
- **YAC Efficiency**: Explosive plays and tackle-breaking ability
- **Catch Rate vs Expected**: Performance relative to target difficulty
- **Blocking Quality**: Pass protection and run blocking effectiveness (RB/TE)

### Sample Size & Talent Context (Phase 5)
- **Dampening for Small Samples**: Prevents outliers from limited play counts
- **Talent-Relative Performance**: Adjusts for quality of supporting cast
- **Consistency Rewards**: Values reliable week-to-week performance

### Minimum Activity Thresholds
Filters out noise from limited participation:
- **RB**: 7 carries per game
- **WR**: 1.875 receptions per game
- **TE**: 1.875 receptions per game
- **QB**: 14 pass attempts per game

---

## Output & Analysis

### Metrics Provided
- **Adjusted Score**: Context-weighted total contribution
- **Avg/Game**: Per-game impact accounting for all adjustments
- **Difficulty**: Average situational difficulty faced (1.000 = neutral, >1.000 = harder)
- **Consistency**: Floor (worst games), Ceiling (best games), Typical (median)
- **Trend**: Performance trajectory (Increasing/Stable/Decreasing)
- **Notable Games**: Performances >150% of typical output

### Available Years
- **2000-2025**: Basic analysis with core multipliers
- **2016-2025**: Full participation data with defenders in box, coverage types, and personnel groupings

---

## Technical Foundation

### Data Sources

**Primary Data (nflverse/nflreadpy):**
- Play-by-play data (2000-present)
- Player statistics (2000-present)
- Team statistics (2000-present)
- Participation data (2016-present)
- NextGen Stats pressure data (2016-present)

**Enhanced Data (FTN Charting - 2022-present):**

FTN (Football Technology Network) provides human-charted flags for specific play characteristics that enable more sophisticated contextual adjustments:

**QB Context Flags:**
- **Play action** vs standard dropback (defense fooled by fake)
- **In pocket** vs **out of pocket** (scrambling/improvisation)
- **Number of blitzers** (pressure from extra rushers)
- **Screen passes** (designed high-percentage throws)

**WR/TE Context Flags:**
- **Contested catches** (defender within 1 yard at catch point)
- **True drops** (catchable ball that hit hands but not caught)
- **Screen passes** (catches in designed space)

**RB Context Flags:**
- **RPO plays** (Run-Pass Option with numbers advantage)
- **Defenders in box** (verified count, more accurate than inferred)
- **Screen passes** (designed space/easier yards)

These objective flags allow us to separate player skill from scheme/situation without subjective grading.

**Language:** Python with Polars for data processing  
**Cache System:** Pre-processed play-by-play data with all multipliers calculated
