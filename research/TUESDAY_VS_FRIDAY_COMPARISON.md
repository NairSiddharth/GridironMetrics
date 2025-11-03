# Tuesday vs Friday Lines Comparison - Player Props Backtest

**Test Date:** November 3, 2025
**Test Period:** 2024 Weeks 5-10
**System:** Tier 2 (recency weighting + career regression + ±17.5% adjustments)
**Hypothesis:** Tuesday lines should be less efficient than Friday lines (more edge opportunity)

---

## Executive Summary

**Result:** Tuesday lines show MARGINALLY better performance (+0.9 pp win rate, +1.9 pp ROI) but NOT enough to be profitable.

**Conclusion:** Early-week betting is NOT the solution. Both Tuesday and Friday lines are too efficient to beat with our current approach.

---

## Performance Comparison Table

| Metric | Tuesday Lines | Friday Lines | Difference |
|--------|---------------|--------------|------------|
| **Total Bets** | 339 | 719 | -380 (-53%) ❌ |
| **Win Rate** | 38.3% | 37.4% | +0.9 pp ✓ |
| **OVER Win Rate** | 37.4% | 38.2% | -0.8 pp |
| **UNDER Win Rate** | 39.4% | 36.6% | +2.8 pp ✓ |
| **ROI** | -32.9% | -34.8% | +1.9 pp ✓ |
| **Total Wagered** | $33,900 | $71,900 | -$38,000 |
| **Total Loss** | -$11,160 | -$25,025 | +$13,865 ✓ |
| **MAE** | 33.98 | 24.60 | +9.38 ❌ |
| **RMSE** | 51.34 | 41.88 | +9.46 ❌ |
| **MAPE** | 59.2% | 66.6% | -7.4 pp ✓ |

**Break-even Win Rate:** 52.4% at -110 odds

---

## Detailed Analysis

### 1. Win Rate: Marginally Better, Still Unprofitable

**Tuesday: 38.3% (130/339)**
- OVER: 37.4% (67/179)
- UNDER: 39.4% (63/160)

**Friday: 37.4% (269/719)**
- OVER: 38.2% (136/356)
- UNDER: 36.6% (133/363)

**Improvement: +0.9 percentage points**

**Analysis:**
- Marginal improvement, but still 14.1 pp below break-even (52.4%)
- Tuesday UNDER bets perform better (+2.8 pp), but OVER bets perform worse (-0.8 pp)
- No consistent pattern suggesting Tuesday lines are systematically softer

### 2. Fewer Bets Available on Tuesday

**Tuesday: 339 bets**
**Friday: 719 bets**
**Difference: -380 bets (-53% fewer opportunities)**

**Analysis:**
- Many players don't have Tuesday lines posted (not all books release early)
- Fewer betting opportunities = less volume = harder to achieve statistical significance
- Even if Tuesday had 42% win rate (hypothetically), only 339 bets vs 719 limits profit potential

### 3. ROI: Better, But Still Catastrophic

**Tuesday: -32.9% ROI**
**Friday: -34.8% ROI**
**Improvement: +1.9 percentage points**

**Tuesday P&L:**
- Wagered: $33,900 (339 bets × $100)
- Lost: $11,160
- ROI: -32.9%

**Friday P&L:**
- Wagered: $71,900 (719 bets × $100)
- Lost: $25,025
- ROI: -34.8%

**Analysis:**
- Both are disastrous ROI figures
- To break even at -110 odds, need +5% ROI minimum
- We're 38-40 percentage points away from break-even

### 4. Projection Accuracy: WORSE on Tuesday

**Tuesday:**
- MAE: 33.98
- RMSE: 51.34
- MAPE: 59.2%

**Friday:**
- MAE: 24.60
- RMSE: 41.88
- MAPE: 66.6%

**Analysis (Critical Finding):**
- Tuesday projections are LESS accurate (MAE +9.38, RMSE +9.46)
- This is counterintuitive - if Tuesday lines were softer, our projections should be MORE accurate
- Possible explanation: Tuesday lines have MORE variance due to injury uncertainty, lineup changes

**Why This Matters:**
- If Tuesday lines were truly inefficient, we'd expect:
  - Higher win rate ✓ (marginal)
  - Better projection accuracy ❌ (worse)
  - Larger edge on winning bets ❌ (similar)

The fact that projections are LESS accurate suggests we're not finding real inefficiency - we're just getting lucky on a smaller sample with higher variance.

### 5. Confidence Grade Performance

#### Tuesday Lines

| Grade | Win Rate | Bets | Expected |
|-------|----------|------|----------|
| **A (15%+ edge)** | 39.0% | 249 | ~62% ❌ |
| **B (12-15% edge)** | 37.8% | 37 | ~58% ❌ |
| **C (8-12% edge)** | 35.8% | 53 | ~54% ❌ |

#### Friday Lines

| Grade | Win Rate | Bets | Expected |
|-------|----------|------|----------|
| **A (15%+ edge)** | 35.3% | 558 | ~62% ❌ |
| **B (12-15% edge)** | 38.9% | 72 | ~58% ❌ |
| **C (8-12% edge)** | 49.4% | 89 | ~54% ✓ |

**Analysis:**
- Tuesday Grade A bets: 39.0% (slightly better than Friday's 35.3%)
- Still FAR below expected 62% if edge were real
- No confidence grade consistently beats break-even on either day

### 6. Bet Volume by Day

**Tuesday:** 339 bets with 8%+ calculated edge
**Friday:** 719 bets with 8%+ calculated edge

**Why Fewer Tuesday Bets?**
1. **Fewer players posted** - Not all sportsbooks release Tuesday lines for all props
2. **Line discrepancies** - Tuesday lines may have different spreads that don't trigger our 8% threshold
3. **Data availability** - Some player injury/lineup info not finalized by Tuesday

**Implication:**
Even if Tuesday had a 45% win rate (hypothetically profitable), only 339 bets/week means:
- Limited absolute profit potential
- Harder to achieve statistical confidence
- More vulnerable to variance over small samples

---

## Statistical Significance Analysis

### Tuesday Results (339 bets, 38.3% win rate)

**95% Confidence Interval:**
- Standard error: sqrt(0.383 × 0.617 / 339) = 0.0264
- Margin of error: 1.96 × 0.0264 = 0.0518
- **CI: [33.1%, 43.5%]**

**Friday Results (719 bets, 37.4% win rate)**

**95% Confidence Interval:**
- Standard error: sqrt(0.374 × 0.626 / 719) = 0.0180
- Margin of error: 1.96 × 0.0180 = 0.0353
- **CI: [33.9%, 40.9%]**

**Overlap:** Confidence intervals overlap significantly [33.9% - 40.9%], meaning the difference is NOT statistically significant.

**Z-test for difference in proportions:**
- Pooled proportion: (130 + 269) / (339 + 719) = 0.377
- SE of difference: sqrt(0.377 × 0.623 × (1/339 + 1/719)) = 0.0313
- Z-score: (0.383 - 0.374) / 0.0313 = 0.29
- **P-value: 0.77 (not significant)**

**Conclusion:** The 0.9 pp improvement on Tuesday is well within random variance. We cannot conclude Tuesday lines are easier to beat.

---

## Why Tuesday Didn't Provide Edge

### Theory vs. Reality

**We Expected (Tuesday Advantage):**
1. Less sharp money has bet the lines
2. Injury reports incomplete (uncertainty = opportunity)
3. Weather forecasts less certain
4. Fewer analysts have adjusted projections
5. Line hasn't been "sharpened" by Wed-Fri betting action

**What We Found:**
1. ✅ Marginally better win rate (+0.9 pp) - but not significant
2. ❌ Worse projection accuracy (MAE +9.38)
3. ❌ Still far from break-even (38.3% vs 52.4% needed)
4. ❌ Much fewer bets available (339 vs 719)
5. ❌ No consistent edge in high-confidence bets

### Why Tuesday Isn't Better

**Hypothesis 1: Market Makers Already Efficient**
- Professional oddsmakers set Tuesday lines using similar models to ours
- They account for uncertainty by widening the spread, not mis-pricing the line
- By the time retail bettors see Tuesday lines, they're already sharp

**Hypothesis 2: Higher Variance, Not True Edge**
- Tuesday lines have more uncertainty (injuries, weather, lineups)
- This creates higher variance outcomes, not predictable inefficiency
- We see slightly better win rate due to luck, not skill

**Hypothesis 3: Selection Bias**
- Only certain props get Tuesday lines (higher profile players)
- These are the MOST efficient props (most betting action)
- We're selecting into the hardest market segment

---

## Implications

### What Tuesday Results Tell Us

1. **Line Timing Doesn't Matter**
   - Tuesday (38.3% WR) ≈ Friday (37.4% WR)
   - Both are far from profitable
   - Early-week betting is NOT the solution

2. **Market Efficiency is Real**
   - Even with 3 fewer days of information, lines are sharp
   - Our model doesn't find edge on Tuesday OR Friday
   - Public data + good methodology ≠ beating the market

3. **Volume Trade-off**
   - Tuesday has 53% fewer bets available
   - Even if marginally better, limited profit opportunity
   - Better to focus on improving model vs. line timing

4. **Projection Accuracy is Key Issue**
   - MAE of 24-34 yards is too large for props
   - Variance in projections exceeds any timing advantage
   - Need fundamentally better predictions, not just better timing

---

## Alternative Strategies to Consider

### Strategy 1: Combine Tuesday and Friday

**Approach:**
- Bet Tuesday lines when available (339 bets)
- Bet Friday lines for remaining props (380 additional bets)
- Total: ~719 bets with weighted average of both

**Expected Performance:**
- Win rate: ~37.8% (weighted average)
- ROI: ~-34% (weighted average)
- Still unprofitable

**Verdict:** ❌ Doesn't solve the problem

### Strategy 2: Wait for Line Movement

**Approach:**
- Scrape Tuesday lines
- Scrape Friday lines
- Only bet when line moved in our favor (Tuesday line was better)

**Example:**
- Tuesday: Patrick Mahomes passing yards 280.5
- Friday: Patrick Mahomes passing yards 285.5
- If we projected 290, Tuesday OVER has more value

**Challenges:**
- Need historical line movement data
- Line movement may be information (injury news), not inefficiency
- Only works if Tuesday lines are systematically mispriced

**Verdict:** ⚠️ Worth testing, but requires line movement data

### Strategy 3: Abandon Line Timing Entirely

**Approach:**
- Accept that line timing doesn't matter
- Focus on improving model fundamentals:
  - Better baselines
  - Better features
  - Machine learning

**Verdict:** ✅ This is the path forward

---

## Recommendations

### Immediate Actions

1. **Stop testing line timing variations**
   - We've now tested Tuesday and Friday
   - No meaningful difference
   - Time to move on

2. **Focus on model quality**
   - Current MAE: 24-34 yards (too high)
   - Current edge calculation: fundamentally broken
   - Need better predictions, not better timing

3. **Investigate machine learning approach**
   - Rule-based system has plateaued at 37-38% win rate
   - ML may capture non-linearities we're missing
   - Worth investing 2-3 weeks to build and test

### Long-Term Strategy

**Option A: Machine Learning (Recommended)**
- Build XGBoost model with 50+ features
- Train on 2020-2023 historical data
- Target: 45-50% win rate as intermediate goal
- Timeline: 3 weeks to build + test

**Option B: Find Specific Niches**
- Focus on backup QBs (less liquidity = less efficient)
- Focus on early season weeks 1-4 (less data = more uncertainty)
- Focus on specific player types where we have edge
- Timeline: 1 week per niche to validate

**Option C: Pivot to DFS**
- Accept betting markets are unbeatable
- Use projection system for daily fantasy sports
- DFS pricing less efficient than betting lines
- Timeline: 1 week to adapt system

---

## Conclusion

**Tuesday Lines Results:** Disappointing but informative

**Key Findings:**
1. Tuesday win rate (38.3%) ≈ Friday win rate (37.4%)
2. Difference is not statistically significant (p=0.77)
3. Tuesday has worse projection accuracy (MAE +9.38)
4. Tuesday has 53% fewer betting opportunities
5. Both days are far from profitable (-33% to -35% ROI)

**Verdict:** Line timing is NOT the issue. Model quality is the issue.

**Next Steps:**
1. Document this comparison ✅
2. Discuss ML implementation approach in detail
3. Make decision: ML, Niches, or Pivot to DFS

**Reality Check:** We've now tested 7 different approaches:
- ❌ Remove dampening (Tier 1)
- ❌ Wider adjustments (Tier 1)
- ❌ Recency weighting (Tier 2)
- ❌ Career regression (Tier 2)
- ❌ Moderate adjustments (Tier 2)
- ❌ Fixed adjustment bugs (Tier 2)
- ❌ Tuesday lines (this test)

None have produced >40% win rate. Time for a fundamentally different approach.