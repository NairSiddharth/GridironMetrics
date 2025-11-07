# Tier 2 Fixes Results - Player Props Backtest

**Test Date:** November 3, 2025
**Test Period:** 2024 Weeks 5-10
**Critical Discovery:** Previous "Tier 2" test was INVALID - all contextual adjustments were silently failing

**Fixes Applied:**
1. **Fixed CACHE_DIR import bug** - All 7 adjustment functions were failing silently
2. **Built schedules cache** - Opponent matchup data was completely missing
3. **Implemented recency weighting** - Last 3 games: 1.5x, games 4-6: 1.0x, games 7+: 0.75x
4. **Added career mean regression** - 80% current season + 20% career average (3-year lookback)
5. **Reduced adjustment ranges** - ±17.5% (down from ±25% in Tier 1)

---

## Results Summary

### Performance Comparison Table

| Metric | Baseline | Tier 1 | Tier 2 | Change (T1→T2) |
|--------|----------|--------|--------|----------------|
| **Total Bets** | 750 | 728 | 719 | -9 bets |
| **Win Rate** | 36.5% | 38.3% | **37.4%** | **-0.9 pp ❌** |
| **OVER Win Rate** | 39.0% | 40.8% | 38.2% | -2.6 pp |
| **UNDER Win Rate** | 35.4% | 35.8% | 36.6% | +0.8 pp |
| **ROI** | -36.2% | -33.3% | **-34.8%** | **-1.5 pp ❌** |
| **Total Wagered** | $75,000 | $72,800 | $71,900 | -$900 |
| **Total Profit** | -$27,152 | -$24,266 | **-$25,025** | **-$759 ❌** |
| **MAE** | 25.14 | 24.26 | 24.60 | +0.34 |
| **RMSE** | 43.48 | 42.37 | 41.88 | -0.49 ✓ |
| **MAPE** | 56.8% | 67.9% | 66.6% | -1.3 pp ✓ |
| **OVER/UNDER Ratio** | 1:2.1 | 1.05:1 | 1.02:1 | More balanced ✓ |

**Break-even Win Rate:** 52.4% at -110 odds

---

## Detailed Analysis

### What Went WRONG

**Tier 2 performed WORSE than Tier 1 despite fixing critical bugs:**

1. **Win Rate Declined**
   - Tier 1: 38.3% win rate
   - Tier 2: 37.4% win rate (-0.9 percentage points)
   - **Gap to break-even INCREASED from -14.1 pp to -15.0 pp**

2. **ROI Worsened**
   - Tier 1: -33.3% ROI
   - Tier 2: -34.8% ROI (-1.5 percentage points)
   - Lost an additional $759 compared to Tier 1

3. **OVER Bets Especially Hurt**
   - OVER win rate dropped from 40.8% to 38.2% (-2.6 pp)
   - Suggests recency weighting + career regression may be under-projecting high-variance players

### What Went RIGHT (Minor)

1. **RMSE Improved Slightly**
   - 42.37 → 41.88 (-0.49 improvement)
   - Indicates slightly less extreme errors

2. **MAPE Improved Marginally**
   - 67.9% → 66.6% (-1.3 pp)
   - Still very poor absolute performance

3. **Maintained OVER/UNDER Balance**
   - 356 OVER vs 363 UNDER (1.02:1 ratio)
   - No systematic bias

### Confidence Grade Performance

| Grade | Win Rate | Bets | Expected Win Rate |
|-------|----------|------|-------------------|
| **A (15%+ edge)** | 35.3% | 558 | ~62% ❌ |
| **B (12-15% edge)** | 38.9% | 72 | ~58% ❌ |
| **C (8-12% edge)** | 49.4% | 89 | ~54% ✓ |

**Paradox Alert:** Grade C (lowest edge) has HIGHEST win rate (49.4%)
- Grade A (highest edge) has LOWEST win rate (35.3%)
- This is completely backwards and indicates edge calculation is fundamentally broken

---

## Root Cause Analysis

### Why Did Tier 2 Fail?

**Theory 1: Recency Weighting Amplifies Noise**
- Emphasizing last 3 games (1.5x weight) may overreact to small samples
- Short-term variance ≠ long-term skill
- Example: WR has 3 great games due to opponent injuries → baseline inflated → loses edge

**Theory 2: Career Mean Regression Anchors to Wrong Baseline**
- Using 3-year career average may not reflect current role/team/scheme
- 20% weight on outdated data dilutes current performance signal
- Example: QB changed offensive coordinator → career data irrelevant

**Theory 3: Reduced Adjustment Ranges Insufficient**
- Tier 1 used ±25%, Tier 2 used ±17.5%
- May have overcorrected - adjustments now too conservative
- Can't overcome baseline errors even when matchup is extreme

**Theory 4: Market Lines Are Simply Too Efficient**
- We're comparing to Friday closing lines (very sharp)
- Sharp bettors have already incorporated all public information
- No amount of baseline/adjustment tuning will beat efficient markets

---

## Comparison to Tier 1 Analysis

### Tier 1 Recommendations (What We Tried)

From [TIER1_FIXES_RESULTS.md](TIER1_FIXES_RESULTS.md):

> **Tier 2: Baseline Methodology Fixes** (CRITICAL)
> 1. Add Recency Weighting ✅ IMPLEMENTED
> 2. Regression to Career Mean ✅ IMPLEMENTED
> 3. Reduce Adjustment Ranges ✅ IMPLEMENTED (±17.5%)

**Result:** All three recommendations were implemented. Performance got WORSE.

**Conclusion:** The Tier 1 analysis misidentified the root cause. Baseline methodology is NOT the primary issue.

---

## The Real Problem

### Edge Calculation is Fundamentally Broken

**Evidence:**
1. **Inverse Correlation Between Edge and Win Rate**
   - Highest edge bets (Grade A, 15%+) win at 35.3%
   - Lowest edge bets (Grade C, 8-12%) win at 49.4%
   - **Real edge would show positive correlation**

2. **All "Edge" is Noise, Not Signal**
   - System identifies 719 bets with 8%+ calculated edge
   - Win rate: 37.4% (need 52.4%)
   - If edge were real, we'd be closer to 55-60% win rate

3. **Projection Errors Are Too Large**
   - MAE of 24.60 on lines averaging ~50-250
   - MAPE of 66.6% (projections off by 2/3 on average)
   - Can't identify edge with ±24 yard uncertainty

### Why Traditional Approaches Won't Work

**We've now tested:**
- ❌ Removing sample size dampening (Tier 1)
- ❌ Widening adjustments (Tier 1)
- ❌ Recency weighting (Tier 2)
- ❌ Career mean regression (Tier 2)
- ❌ Moderate adjustments (Tier 2)

**All five approaches failed to improve win rate above 38.3%**

This suggests the problem is not methodological - it's **structural**.

---

## What We're Missing

### Likely Missing Factors

1. **Teammate Quality**
   - QB performance depends on O-line, receivers, playcaller
   - RB performance depends on O-line, blocking scheme
   - We don't model any of this

2. **Game Script / Pace**
   - Teams trailing heavily pass more → inflates QB/WR stats
   - Teams leading heavily run more → inflates RB stats
   - We don't predict game script

3. **Injury Context**
   - Player at 80% health performs differently than 100%
   - Backup players get elevated roles when starters injured
   - We don't model injury impact on playing time/usage

4. **Market-Moving Information**
   - Sharp bettors know about lineup changes, weather, coaching decisions
   - We're using Friday lines (all this info already priced in)
   - We'd need Tuesday lines or earlier to find inefficiency

5. **Usage / Target Share**
   - Not all games are equal - player usage varies by game plan
   - Red zone role, slot vs outside, formation packages
   - We use season-long averages, missing week-to-week variation

---

## Tier 3 Strategy (Last Resort)

### Option A: Machine Learning Approach

**Replace rule-based adjustments with ML model:**
- Train on historical player props data (2020-2023)
- Features: baseline stats, opponent metrics, weather, injuries, vegas totals
- Target: Actual outcome vs betting line
- Model: XGBoost or neural network

**Pros:**
- Can discover non-linear relationships
- Automatically weighs features by importance
- May capture interactions we're missing

**Cons:**
- Requires significant data engineering
- Risk of overfitting to historical data
- Markets may have evolved since training period

### Option B: Focus on Player-Specific Inefficiencies

**Instead of system-wide edge, find niche opportunities:**
- Identify specific players where lines are consistently wrong
- Focus on backup QBs/RBs (less betting liquidity = less efficient)
- Target early-season props (less data = more uncertainty)

**Pros:**
- Smaller sample size = easier to validate
- Can manually research specific situations
- May find exploitable patterns

**Cons:**
- Limited betting opportunities
- Requires constant monitoring
- Patterns may disappear quickly once discovered

### Option C: Abandon This Approach Entirely

**Accept that Friday closing lines are too efficient to beat with public data:**
- Sharp bettors have everything we have, plus:
  - Proprietary injury reports
  - Lineup change information
  - Weather updates
  - Betting market flow data

**Alternative:** Use this system for DFS (daily fantasy) instead of sports betting
- DFS pricing less efficient than betting lines
- Salary constraints create opportunities
- Don't need to beat vig, just need to beat other DFS players

---

## Recommendations

### Immediate Next Steps

**DO NOT proceed with Tier 3 baseline improvements.**

Instead:

1. **Test on 2023 Season**
   - Cross-validate: Does Tier 2 system perform similarly poorly on 2023?
   - If yes: Problem is methodological (can potentially fix)
   - If no: 2024 was an unusual season (less fixable)

2. **Test on Weeks 1-4 (Early Season)**
   - Theory: Early season has less data = more uncertainty = less efficient lines
   - If win rate improves: Focus on early-season props only
   - If no change: Confirms lines are efficient across all weeks

3. **Compare to Market Consensus**
   - Scrape multiple sportsbooks' lines
   - Find props where books disagree significantly
   - Test if our edge aligns with line discrepancies

4. **Analyze Individual Player Performance**
   - Which specific players do we project well vs poorly?
   - Are errors random or systematic (e.g., always overproject mobile QBs)?
   - Can we blacklist problematic player types?

### Long-Term Decision Point

After tests 1-4 above:

**If win rate remains <42% across all scenarios:**
- **Accept that public data cannot beat closing lines**
- **Pivot to DFS or abandon sports betting props**

**If win rate improves in specific scenarios (e.g., early season, 2023, certain positions):**
- **Narrow focus to those specific opportunities**
- **Build specialized models for those niches**

---

## Key Learnings

1. **Fixing bugs ≠ Improving performance**
   - We fixed critical bugs (CACHE_DIR, schedules)
   - Adjustments are now working properly
   - Performance still got worse

2. **More sophisticated baselines ≠ Better predictions**
   - Recency weighting is theoretically sound
   - Career mean regression prevents overfitting
   - Both hurt performance in practice

3. **Edge calculations are unreliable**
   - Grade A bets (15%+ edge) win at 35.3%
   - This should be mathematically impossible if edge were real
   - Our edge = noise, not signal

4. **Market efficiency is underestimated**
   - We assumed public data + good methodology = edge
   - Markets price in public data immediately
   - Need private information or better models than entire market

---

## Conclusion

**Tier 2 Results:** Worse than Tier 1, confirming fundamental issues

**Status:** 37.4% win rate, -34.8% ROI (-$25,025 loss)

**Reality Check:** We've now tested 5 different approaches across Tier 1 and Tier 2. None improved win rate beyond 38.3%. The gap to break-even (52.4%) remains enormous.

**Recommendation:** **STOP trying to beat Friday closing lines with rule-based public data.** Either:
1. Invest in ML approach with much more data/features
2. Find niche inefficiencies (early season, backup players)
3. Pivot to DFS where pricing is less efficient
4. Accept this won't work and move on

Beating betting markets is one of the hardest problems in data science. We've built a sophisticated system with recency weighting, career regression, 7 contextual adjustments, and injury modeling. It's not enough.

The market is smarter than our model. Time to either fundamentally change the approach or move on.