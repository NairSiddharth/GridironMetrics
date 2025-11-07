# Tier 1 Fixes Results - Player Props Backtest

**Test Date:** November 3, 2025
**Test Period:** 2024 Weeks 5-10
**Fixes Applied:**
1. Removed sample size dampening from prop projections
2. Widened opponent defense adjustments (±15% → ±25%)
3. Widened success rate adjustments (±15% → ±20%)

---

## Results Comparison

### Before Tier 1 Fixes (Baseline)

**Configuration:**
- Sample size dampening: APPLIED (reducing projections by 15-30%)
- Opponent defense range: ±15% (0.85x to 1.15x)
- Success rate range: ±15% (0.92x to 1.15x)

**Performance:**
- **Total Bets:** 750 (meeting 8% edge threshold)
- **Win Rate:** 36.5% (274/750 wins)
  - OVER bets: 39.0% (94/241)
  - UNDER bets: 35.4% (180/509)
- **ROI:** -36.2%
- **Money:** -$27,152 on $75,000 wagered
- **Projection Accuracy:**
  - MAE: 25.14
  - RMSE: 43.48
  - MAPE: 56.8%

**Key Issue:** Severe UNDER bias (509 UNDER vs 241 OVER bets = 2.1:1 ratio)

---

### After Tier 1 Fixes

**Configuration:**
- Sample size dampening: REMOVED
- Opponent defense range: ±25% (0.75x to 1.25x)
- Success rate range: ±20% (0.80x to 1.20x)

**Performance:**
- **Total Bets:** 728 (meeting 8% edge threshold)
- **Win Rate:** 38.3% (279/728 wins) **[+1.8 pp]**
  - OVER bets: 40.8% (152/373) **[+1.8 pp]**
  - UNDER bets: 35.8% (127/355) **[+0.4 pp]**
- **ROI:** -33.3% **[+2.9 pp improvement]**
- **Money:** -$24,266 on $72,800 wagered **[+$2,886 improvement]**
- **Projection Accuracy:**
  - MAE: 24.26 **[-0.88 improvement]**
  - RMSE: 42.37 **[-1.11 improvement]**
  - MAPE: 67.9% **[-11.1 pp worse]**

**Key Improvement:** Better OVER/UNDER balance (373 OVER vs 355 UNDER = 1.05:1 ratio)

---

## Detailed Analysis

### Improvements Observed

1. **Balanced Projection Distribution** ✅
   - Before: 2.1:1 UNDER/OVER ratio (severe bias)
   - After: 1.05:1 UNDER/OVER ratio (nearly balanced)
   - **Impact:** Removing dampening eliminated systematic underestimation

2. **Marginal Win Rate Improvement** ⚠️
   - Overall: +1.8 percentage points (36.5% → 38.3%)
   - OVER bets: +1.8 pp (39.0% → 40.8%)
   - UNDER bets: +0.4 pp (35.4% → 35.8%)
   - **Impact:** Small but measurable improvement

3. **Slightly Better Projection Accuracy** ⚠️
   - MAE improved by 0.88 (25.14 → 24.26)
   - RMSE improved by 1.11 (43.48 → 42.37)
   - **Impact:** Marginal improvement in raw accuracy

### Disappointing Results

**Expected vs. Actual:**
- **Expected:** 48-52% win rate (based on analysis)
- **Actual:** 38.3% win rate
- **Gap:** -10 to -14 percentage points below expectation

**Why the Fixes Didn't Work as Expected:**

1. **Sample Size Dampening Was NOT the Primary Issue**
   - Removing it fixed the UNDER bias but didn't improve win rate significantly
   - The 1.8pp improvement is far short of the predicted 12-16pp jump
   - Conclusion: Dampening was a secondary problem, not the primary cause

2. **Wider Adjustments May Be Overcompensating**
   - MAPE worsened from 56.8% to 67.9% (+11.1pp)
   - This suggests adjustments might be TOO aggressive now
   - Overshoot in both directions = more extreme errors

3. **Fundamental Baseline Issue Still Present**
   - MAE of 24.26 means projections are still off by ~24 yards/receptions on average
   - This is still a large error for props (e.g., 24 yards on a 250-yard line = 10% error)
   - Even with balanced bias, projections aren't accurate enough

---

## Root Cause Re-Analysis

### The Real Problem: Baseline Calculation Methodology

**Current baseline:** Weighted rolling average through week N-1
- Game 1: 0.25x weight
- Game 2: 0.50x weight
- Game 3: 0.75x weight
- Game 4+: 1.0x weight

**Issue Identified:**

1. **Recency Bias Missing**
   - All games weighted equally after game 4
   - Game from week 5 weighted same as game from week 10
   - Recent performance trends ignored

2. **No Regression to Mean**
   - Hot streaks treated as new baseline
   - Cold streaks treated as new baseline
   - No dampening toward player's career norms

3. **Context-Free Baseline**
   - Doesn't account for strength of schedule
   - Doesn't account for injury history
   - Doesn't account for role changes

### Why Adjustments Can't Fix Bad Baselines

**Current flow:**
```
Bad Baseline → Apply Adjustments → Still Bad Projection
```

**Example:**
- Player's true average: 70 receiving yards
- Had 3 bad games (40, 35, 45 yards) → Baseline: 40 yards
- Apply +20% adjustment for elite matchup: 40 × 1.20 = 48 yards
- Betting line: 65 yards
- **System flags:** 48 vs 65 = 26% edge for UNDER
- **Reality:** Player returns to form with 75 yards → LOSS

**Root issue:** Baseline (40 yards) is too low, adjustment can't overcome it

---

## Revised Improvement Strategy

### What Didn't Work

❌ **Tier 1 Fixes (Implemented):**
- Removing sample size dampening: Minimal impact (+1.8pp)
- Wider adjustment ranges: Possible overcorrection (MAPE worsened)

### What We Need to Try Next

### **Tier 2: Baseline Methodology Fixes** (CRITICAL)

1. **Add Recency Weighting**
   - Last 3 games: 1.5x weight
   - Games 4-6: 1.0x weight
   - Games 7+: 0.75x weight
   - Captures recent trends and form

2. **Regression to Career Mean**
   - Blend current season average with 3-year career average
   - Formula: `0.75 × current_season + 0.25 × career_average`
   - Prevents overreaction to small samples

3. **Strength-of-Schedule Adjustment on Baseline**
   - Adjust baseline for difficulty of past opponents
   - Player facing tough schedule → inflate baseline slightly
   - Player facing weak schedule → deflate baseline slightly

4. **Reduce Adjustment Ranges Back to Moderate**
   - Current ±25% may be too aggressive
   - Try ±18-20% instead
   - Let better baselines do the heavy lifting

### **Tier 3: Use Prior Season Data for Early Weeks**

- Week 1-2: Blend 60% prior season + 40% current
- Week 3-4: Blend 40% prior season + 60% current
- Week 5+: 100% current season

This was already in the plan but deferred. Results suggest it's more critical than anticipated.

### **Tier 4: Investigate Market Line Quality**

- Are we comparing to closing lines or opening lines?
- Closing lines are sharp; opening lines have more inefficiency
- If using closing lines, we're competing with sharp money
- May need to scrape earlier lines for better edge opportunities

---

## Recommendations

### Immediate Next Steps

1. **Implement Tier 2 fixes (Baseline improvements):**
   - Add recency weighting to rolling average
   - Add regression to career mean
   - Reduce adjustment ranges to ±18-20%

2. **Re-test on weeks 5-10:**
   - Target: 45-50% win rate (intermediate goal)
   - If achieved, proceed to Tier 3

3. **If Tier 2 fails (<42% win rate):**
   - Investigate market line timing (closing vs opening)
   - Consider whether 2024 was an unusual season
   - Test on 2023 data for cross-validation

### Long-term Strategy

**Do NOT deploy to live betting until:**
- ✅ Win rate ≥52% validated across multiple weeks
- ✅ ROI ≥+5% with realistic kelly sizing
- ✅ Cross-validated on 2023 and 2024 seasons
- ✅ Tested on weeks 1-4 with prior season data

**Current status:** System is NOT profitable. Tier 1 fixes provided marginal improvement but did not solve the fundamental problem.

---

## Conclusion

**Tier 1 Results:** Disappointing but informative

**Key Learnings:**
1. Sample size dampening was a symptom, not the disease
2. Baseline calculation methodology is likely the root cause
3. Adjustments can't fix bad baselines
4. Need to focus on baseline quality before fine-tuning adjustments

**Next Iteration:** Implement Tier 2 baseline improvements and retest.

**Reality Check:** Beating betting markets is extremely difficult. A 38.3% win rate suggests we're not finding real edge—we're just identifying noise. The path forward requires fundamentally rethinking how we calculate baselines, not just tweaking multipliers.
