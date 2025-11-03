# NFL Betting Data Sources - Research Findings

## Data Sources Comparison

### Source 1: Historical Archive (cache/nfl_archive_10Y.json)
**Coverage:** 2011-2021 (11 years, 2,956 games)

**Data Format:**
```json
{
  "season": 2011,
  "date": 20110908.0,
  "home_team": "Packers",
  "away_team": "Saints",
  "home_open_spread": -4.5,
  "away_open_spread": 4.5,
  "home_close_spread": -5.0,
  "away_close_spread": 5.0,
  "home_close_ml": -250,
  "away_close_ml": 210,
  "open_over_under": 46.0,
  "close_over_under": 48.0,
  "home_2H_spread": -24.5,
  "away_2H_spread": 24.5,
  "2H_total": 0.0,
  "home_final": "42",
  "away_final": "34"
}
```

**Available Markets:**
- Opening spreads (both home/away)
- Closing spreads (both home/away)
- Moneylines (close only)
- Totals (open/close)
- Second half spreads and totals
- Final scores and quarterly breakdowns

**Quality:**
- ✅ Consistent coverage: ~267 games/year
- ✅ Both opening AND closing lines
- ✅ Simple, flat structure
- ✅ Ready to use
- ⚠️ Team names use varied formats ("Packers", "St.Louis", "Commanders")
- ⚠️ No specific bookmaker attribution
- ⚠️ Coverage ends 2021 (4 seasons old)

---

### Source 2: The Odds API (Live/Current)
**Coverage:** Current season + historical from mid-2020

**API Costs (Free Tier: 500 credits/month):**
- Current odds: 3 credits per request (h2h + spreads + totals)
- Historical odds: 30 credits per request (10x multiplier)
- Week 9 test: Used 9 credits for 14 games

**Data Format:**
```json
{
  "id": "c2fd8a23091a954fb21ff6d3537db826",
  "sport_key": "americanfootball_nfl",
  "commence_time": "2025-11-03T01:22:51Z",
  "home_team": "Washington Commanders",
  "away_team": "Seattle Seahawks",
  "bookmakers": [
    {
      "key": "draftkings",
      "title": "DraftKings",
      "last_update": "2025-11-03T02:43:52Z",
      "markets": [
        {
          "key": "h2h",
          "outcomes": [
            {"name": "Seattle Seahawks", "price": -6500},
            {"name": "Washington Commanders", "price": 1800}
          ]
        },
        {
          "key": "spreads",
          "outcomes": [
            {"name": "Seattle Seahawks", "price": 100, "point": -21.5},
            {"name": "Washington Commanders", "price": -130, "point": 21.5}
          ]
        },
        {
          "key": "totals",
          "outcomes": [
            {"name": "Over", "price": -110, "point": 59.5},
            {"name": "Under", "price": -120, "point": 59.5}
          ]
        }
      ]
    }
  ]
}
```

**Available Markets:**
- h2h (moneylines)
- spreads (point spreads with juice)
- totals (over/under)
- Multiple bookmakers per game (DraftKings, FanDuel, BetMGM, Caesars, etc.)

**Quality:**
- ✅ Multiple bookmakers (consensus lines)
- ✅ Real-time updates
- ✅ Standardized team names
- ✅ ISO timestamps
- ✅ Detailed bookmaker metadata
- ⚠️ Nested structure (more complex to parse)
- ⚠️ Live odds update during games (need to time API calls correctly)
- ⚠️ Historical data very expensive (30 credits per game)

---

## Credit Cost Analysis

### Current Season (2025)
**Scenario 1: One snapshot per week (Tuesday opening lines)**
- 18 weeks × 1 request = 18 requests × 3 credits = **54 credits**
- Covers all ~288 regular season games
- **Easily fits free tier** ✅

**Scenario 2: One snapshot per game (pre-game lines)**
- 288 games ÷ 14 games/request = ~21 requests × 3 credits = **63 credits**
- More granular, specific to each game
- **Fits free tier** ✅

**Scenario 3: Opening + Closing lines**
- Tuesday opening + Sunday closing = 36 requests × 3 credits = **108 credits**
- Full line movement tracking
- **Fits free tier** ✅

### Historical Backfill (2022-2024)
**Full backfill cost:**
- 2022-2024: ~800 games
- 800 games ÷ 14 games/request = ~57 requests × 30 credits = **1,710 credits**
- Requires: $30/month tier (20K credits) for 1 month

**Selective backfill (key games only):**
- Playoffs only: ~40 games = ~9 requests × 30 credits = **270 credits**
- **Fits free tier** ✅

---

## Data Quality Comparison (Overlap Period: 2020-2021)

### Historical Archive
- 2020: 269 games
- 2021: 284 games
- Total: **553 games**

### The Odds API
- Claims coverage from "mid-2020"
- Historical endpoint available
- Cost: 30 credits per request × ~40 requests = **1,200 credits**

**Recommendation:** Use 50-100 credits to sample 2020 data from The Odds API and compare accuracy with historical archive before committing to full backfill.

---

## Team Name Normalization Issues

### Historical Archive Format:
- "Packers", "Saints", "St.Louis", "Commanders", "Jaguars", "Texans"
- Inconsistent (sometimes city, sometimes team name)
- Needs mapping to nflverse team codes

### The Odds API Format:
- "Green Bay Packers", "New Orleans Saints", "Los Angeles Rams", "Washington Commanders"
- Full official names
- Needs mapping to nflverse team codes

### Solution Required:
Create team name mapping dictionary to normalize both sources to nflverse abbreviations:
```python
{
  "Packers": "GB",
  "Green Bay Packers": "GB",
  "Saints": "NO",
  "New Orleans Saints": "NO",
  ...
}
```

---

## Recommended Standardized Schema

```python
{
  "game_id": str,          # Unique identifier
  "season": int,            # Year
  "week": int,              # Week number
  "date": str,              # ISO format: "2025-11-03"
  "home_team": str,         # nflverse code: "GB", "NO", etc.
  "away_team": str,         # nflverse code

  # Spreads
  "home_open_spread": float,
  "away_open_spread": float,
  "home_close_spread": float,
  "away_close_spread": float,

  # Moneylines
  "home_open_ml": int,      # American odds
  "away_open_ml": int,
  "home_close_ml": int,
  "away_close_ml": int,

  # Totals
  "open_total": float,
  "close_total": float,

  # Metadata
  "source": str,            # "historical_archive" or "odds_api"
  "bookmaker": str,         # "consensus" or specific book

  # Optional: Results (if game completed)
  "home_score": int,
  "away_score": int
}
```

---

## Recommended Cache Structure

```
cache/
  betting_lines/
    historical/              # 2011-2021 from archive
      2011.parquet
      2012.parquet
      ...
      2021.parquet

    current/                 # 2022+ from The Odds API
      2022.parquet
      2023.parquet
      2024.parquet
      2025.parquet

    raw/                     # Raw API responses (for debugging)
      odds_api/
        2025-11-03_opening.json
        2025-11-03_closing.json

    metadata.json            # Coverage info, last update, etc.
```

---

## Findings Summary

### Historical Archive (2011-2021)
**Pros:**
- ✅ 11 years of data ready to use
- ✅ Both opening and closing lines included
- ✅ No API costs
- ✅ Complete regular season + playoffs

**Cons:**
- ⚠️ Outdated (ends 2021)
- ⚠️ Unknown source/bookmaker
- ⚠️ Team name inconsistencies
- ⚠️ No validation possible

**Best Use:** Historical baseline (2011-2021)

---

### The Odds API
**Pros:**
- ✅ Current/live data
- ✅ Multiple bookmakers (consensus)
- ✅ Official, reliable source
- ✅ Real-time updates
- ✅ Affordable for current season tracking

**Cons:**
- ⚠️ Historical data expensive (30x cost)
- ⚠️ Free tier limits deep historical analysis
- ⚠️ Needs careful timing for opening/closing lines

**Best Use:** Current season tracking (2022+), ongoing maintenance

---

## Recommended Implementation Strategy

### Phase 1: Immediate (Free Tier)
1. **Process historical archive** (2011-2021)
   - Normalize team names to nflverse codes
   - Convert to standardized schema
   - Save as parquet files
   - Cost: $0

2. **Track 2025 season** with The Odds API
   - Pull weekly opening lines (Tuesday)
   - ~54 credits total for season
   - Cost: $0 (free tier)

### Phase 2: Validation (50 credits)
1. **Sample 2020 data** from The Odds API
   - Pull ~15-20 games from 2020
   - Compare with historical archive for accuracy
   - Validate data quality
   - Cost: ~50 credits (free tier)

### Phase 3: Optional Backfill (Paid Tier)
1. **If validation passes** and you want 2022-2024 data:
   - Use $30/month tier for 1 month
   - Backfill 2022-2024 (~1,700 credits)
   - Cancel after backfill complete
   - Cost: $30 one-time

2. **Alternative:** Use free tier for playoff games only
   - 2022-2024 playoffs: ~40 games
   - ~270 credits from free tier allowance
   - Cost: $0

---

## Next Steps

1. ✅ **DONE:** Pull sample data from The Odds API
2. ✅ **DONE:** Examine historical archive structure
3. **TODO:** Create team name normalization mapping
4. **TODO:** Write data transformation scripts:
   - Historical archive → standardized format
   - The Odds API → standardized format
5. **TODO:** Set up automated weekly pulls for 2025 season
6. **TODO:** (Optional) Validate historical archive against 2020 API data

---

## Credit Usage Report

**Current Status:**
- Free tier limit: 500 credits/month
- Used today: 9 credits (testing)
- Remaining: 491 credits

**Projected Usage:**
- 2025 season tracking: ~54 credits (opening lines only)
- 2025 season + validation: ~108 credits (opening + closing)
- Buffer for testing/debugging: ~50 credits
- **Total: ~160 credits for full year** (well under limit)
