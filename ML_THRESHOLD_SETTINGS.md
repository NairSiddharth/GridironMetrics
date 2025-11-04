# ML Model Threshold Settings

**Purpose:** Track optimal threshold selections for each prop type based on evaluation results.

**Last Updated:** 2025-11-03

---

## Selected Thresholds

| Prop Type       | Threshold | Accuracy | ROI    | Bets/Week | Status      | Notes                           |
|-----------------|-----------|----------|--------|-----------|-------------|---------------------------------|
| passing_yards   | 60+ yards | 57.1%    | +9.1%  | ~14       | ✅ SELECTED | Good volume with strong ROI     |
| passing_tds     | TBD       | -        | -      | -         | 🔄 PENDING  | Awaiting evaluation results     |
| rushing_yards   | TBD       | -        | -      | -         | ⬜ NOT YET  | Model not trained               |
| rushing_tds     | TBD       | -        | -      | -         | ⬜ NOT YET  | Model not trained               |
| receptions      | TBD       | -        | -      | -         | ⬜ NOT YET  | Model not trained               |
| receiving_yards | TBD       | -        | -      | -         | ⬜ NOT YET  | Model not trained               |
| receiving_tds   | TBD       | -        | -      | -         | ⬜ NOT YET  | Model not trained               |

---

## Threshold Selection Criteria

When evaluating a new prop type, consider:

1. **Break-even Threshold:** 52.4% accuracy minimum (with -110 odds)
2. **Volume Preference:**
   - Conservative: 4-7 bets/week
   - Balanced: 10-15 bets/week
   - Aggressive: 18-25 bets/week
3. **ROI Target:** Aim for +5% or higher for sustainability
4. **Consistency:** Review both 2023 and 2024 performance

---

## Detailed Settings

### passing_yards

**Selected:** 60+ yards

**Rationale:**
- Provides good weekly volume (~14 bets)
- Strong accuracy (57.1%, +4.7% above break-even)
- Solid ROI (+9.1%)
- Slightly more aggressive than peak (70-80 yards) but still very profitable

**Full Range Analysis:**
| Threshold | Accuracy | ROI     | Bets/Week |
|-----------|----------|---------|-----------|
| 30 yards  | 53.2%    | +1.5%   | 22.2      |
| 40 yards  | 54.1%    | +3.1%   | 18.5      |
| 50 yards  | 55.3%    | +5.8%   | 15.9      |
| **60 yards**  | **57.1%**    | **+9.1%**   | **13.9** |
| 70 yards  | 57.8%    | +10.2%  | 12.1      |
| 80 yards  | 58.5%    | +11.7%  | 10.9      | ← Peak ROI
| 90 yards  | 58.3%    | +11.4%  | 8.8       |
| 100 yards | 57.9%    | +10.8%  | 6.9       |

**Alternative Strategies:**
- **Conservative:** 80+ yards (58.5% acc, +11.7% ROI, ~11 bets/week) - Max ROI
- **Aggressive:** 40+ yards (54.1% acc, +3.1% ROI, ~18 bets/week) - Higher volume

---

### passing_tds

**Status:** Model training in progress (ETA: ~10-15 minutes)

**Next Steps:**
1. Wait for training to complete
2. Run evaluation: `python scripts/evaluate_ml_model_multi_year.py` (modify for passing_tds)
3. Review profitable ranges
4. Select threshold based on volume/accuracy/ROI preferences
5. Update this file

**Expected Differences from passing_yards:**
- Smaller threshold increments (TDs are discrete: 0, 1, 2, 3, 4)
- Threshold likely measured in TDs (e.g., "1.0+ TD edge") rather than yards
- Lower overall volumes (TDs are rarer events)
- May see different optimal strategies

---

## Implementation Notes

**For Live Betting Integration:**

When making predictions, only place bets when:
```python
confidence = abs(ml_prediction - betting_line)

# For passing_yards
if confidence >= 60:  # yards
    place_bet(prediction, line)

# For passing_tds (example - TBD after evaluation)
if confidence >= 0.8:  # TDs
    place_bet(prediction, line)
```

**Confidence Calculation:**
- `confidence = |ML_prediction - betting_line|`
- Only bet when confidence exceeds selected threshold for that prop type
- Each prop type has independent threshold (passing_yards uses yards, passing_tds uses TDs, etc.)

---

## Evaluation Results Archive

**Evaluation Date:** 2025-11-03

### passing_yards
- Training data: 2015-2024 (5,124 examples)
- Test years: 2023 & 2024
- Overall accuracy: 53.0%
- Overall ROI: +1.1% (all bets)
- Profitable range: 30-120 yards
- Selected: 60+ yards

*Full report: ML_EVALUATION_RESULTS.md*
