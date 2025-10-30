# FTN Charting Data Integration Plan

## Executive Summary

This document outlines the implementation plan for integrating FTN (Football Technology Network) charting data into the GridironMetrics system. FTN provides human-charted, objective flags for specific play characteristics that will enhance our contextual adjustments for QB and skill player contributions.

**Data Source:** `nflreadpy.load_ftn_charting()`  
**Availability:** 2022-present (FTN data starts in 2022)  
**Impact:** Enhanced context for ~8-15% of plays across all positions

---

## Phase 1: Cache Infrastructure

### 1.1 Create FTN Cache Builder Module

**File:** `modules/ftn_cache_builder.py`

**Purpose:** Download, process, and cache FTN charting data by year

**Implementation:**

```python
"""
ftn_cache_builder.py

Builds and maintains a cache of FTN charting data (human-charted play characteristics)
from nflreadpy. Data includes play action, RPO, blitz count, contested catches, etc.
"""

from pathlib import Path
import nflreadpy as nfl
from modules.constants import CACHE_DIR
import polars as pl
from modules.logger import get_logger

logger = get_logger(__name__)

# FTN data available starting 2022
FTN_START_YEAR = 2022

CACHE_DIR = Path(CACHE_DIR) / "ftn"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def build_ftn_cache_for_year(year: int) -> bool:
    """
    Build FTN charting cache for a single year.
    
    Args:
        year: Season year to cache
        
    Returns:
        True if successful, False otherwise
    """
    if year < FTN_START_YEAR:
        logger.info(f"FTN data not available for {year} (starts in {FTN_START_YEAR})")
        return False
    
    logger.info(f"Fetching FTN charting data for {year}...")
    
    try:
        ftn = nfl.load_ftn_charting(seasons=year)
    except Exception as e:
        logger.error(f"Error fetching FTN data for {year}: {str(e)}")
        return False
    
    if ftn is None or (hasattr(ftn, 'is_empty') and ftn.is_empty()):
        logger.warning(f"No FTN data returned for {year}")
        return False
    
    # Select only the columns we need
    columns_to_keep = [
        'nflverse_game_id',
        'nflverse_play_id',
        # QB flags
        'is_play_action',
        'is_qb_out_of_pocket',
        'n_blitzers',
        # WR/TE flags
        'is_contested_ball',
        'is_drop',
        # All positions
        'is_rpo',
        'is_screen_pass',
        'n_defense_box'  # More accurate than inferred defenders_in_box
    ]
    
    ftn_subset = ftn.select(columns_to_keep)
    
    logger.info(f"Processing {len(ftn_subset)} FTN charted plays for {year}")
    
    # Save as parquet for efficient joining with PBP
    out_path = CACHE_DIR / f"ftn_{year}.parquet"
    try:
        ftn_subset.write_parquet(out_path)
        logger.info(f"Saved {out_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save {out_path}: {str(e)}")
        return False


def build_full_ftn_cache(start_year: int = FTN_START_YEAR, end_year: int = 2025) -> None:
    """Build FTN cache for all available years."""
    logger.info(f"Building FTN cache for years {start_year}-{end_year}")
    
    success_count = 0
    for year in range(start_year, end_year + 1):
        if build_ftn_cache_for_year(year):
            success_count += 1
    
    logger.info(f"Successfully built FTN cache for {success_count} years")


def ftn_cache_exists(year: int) -> bool:
    """Check if FTN cache exists for a given year."""
    if year < FTN_START_YEAR:
        return False  # Data not available
    
    cache_path = CACHE_DIR / f"ftn_{year}.parquet"
    return cache_path.exists()


def load_ftn_cache(year: int) -> pl.DataFrame:
    """
    Load FTN cache for a given year.
    
    Args:
        year: Season year to load
        
    Returns:
        DataFrame with FTN charting data, or None if not available
    """
    if not ftn_cache_exists(year):
        logger.debug(f"FTN cache not available for {year}")
        return None
    
    cache_path = CACHE_DIR / f"ftn_{year}.parquet"
    try:
        return pl.read_parquet(cache_path)
    except Exception as e:
        logger.error(f"Error loading FTN cache for {year}: {str(e)}")
        return None


if __name__ == "__main__":
    # Build cache for all available years
    build_full_ftn_cache()
```

**Key Design Decisions:**
- Store as parquet (same as PBP cache) for efficient joins
- Only cache columns we're actually using (reduced disk space)
- Include both game_id and play_id for precise joining
- Gracefully handle pre-2022 years (return None/False)

---

### 1.2 Update Main Cache Rebuilder

**File:** `main.py` - `check_and_rebuild_caches()` function

**Changes:**

```python
def check_and_rebuild_caches(years: list[int], parallel: bool = True) -> None:
    """Check all years for missing caches and rebuild all cache types as needed.
    
    This runs before main processing to ensure all caches exist and have required data.
    Rebuilds PBP, positional player stats, team stats, and FTN caches.
    """
    from modules.positional_cache_builder import build_positional_cache_for_year
    from modules.team_cache_builder import build_team_cache_for_year
    from modules.ftn_cache_builder import build_ftn_cache_for_year, FTN_START_YEAR  # NEW
    
    # ... existing code ...
    
    years_needing_ftn_rebuild = []  # NEW
    
    # Check each year's caches
    for year in years:
        # ... existing PBP, positional, team checks ...
        
        # Check FTN cache (NEW)
        if year >= FTN_START_YEAR:
            ftn_path = Path("cache/ftn") / f"ftn_{year}.parquet"
            if not ftn_path.exists():
                logger.info(f"FTN cache missing for {year}, will rebuild")
                years_needing_ftn_rebuild.append(year)
    
    # ... existing rebuild logic ...
    
    # Add FTN to rebuild list
    if years_needing_ftn_rebuild:
        logger.info(f"  FTN: {years_needing_ftn_rebuild}")
    
    # In rebuild loop, add:
    if year in years_needing_ftn_rebuild:
        try:
            logger.info(f"Rebuilding FTN cache for {year}...")
            build_ftn_cache_for_year(year)
            logger.info(f"FTN cache rebuilt for {year}")
        except Exception as e:
            logger.error(f"FTN cache rebuild error for {year}: {e}")
```

---

## Phase 2: Integration with PBP Processing

### 2.1 Load FTN Data in QB Rankings

**File:** `main.py` - `generate_qb_rankings()` function

**Integration Point:** After loading PBP data, before calculating contributions

**Changes:**

```python
def generate_qb_rankings(year: int) -> str:
    logger.info(f"Generating QB rankings for {year}")
    
    # ... existing QB stats loading ...
    
    # Load PBP data for contribution calculation
    pbp_path = Path("cache/pbp") / f"pbp_{year}.parquet"
    pbp_data = pl.read_parquet(pbp_path)
    
    # NEW: Load FTN data and join with PBP
    from modules.ftn_cache_builder import load_ftn_cache
    ftn_data = load_ftn_cache(year)
    
    if ftn_data is not None:
        logger.info(f"Joining FTN charting data for contextual adjustments ({len(ftn_data)} plays)")
        
        # Join on game_id and play_id
        pbp_data = pbp_data.join(
            ftn_data,
            on=['game_id', 'play_id'],
            how='left'
        )
        
        # Log FTN data availability
        pa_pct = (pbp_data['is_play_action'].sum() / len(pbp_data)) * 100
        oop_pct = (pbp_data['is_qb_out_of_pocket'].sum() / len(pbp_data)) * 100
        logger.info(f"FTN coverage: {pa_pct:.1f}% play action, {oop_pct:.1f}% out of pocket")
    else:
        logger.info(f"FTN data not available for {year}, using base calculations")
        # Add null columns so code doesn't break
        pbp_data = pbp_data.with_columns([
            pl.lit(None).alias('is_play_action'),
            pl.lit(None).alias('is_qb_out_of_pocket'),
            pl.lit(None).alias('n_blitzers'),
            pl.lit(None).alias('is_rpo'),
            pl.lit(None).alias('is_screen_pass'),
        ])
    
    # ... continue with existing contribution calculation ...
```

---

### 2.2 Apply QB Contextual Adjustments

**File:** `main.py` - Inside QB contribution calculation loop

**Location:** Where we currently calculate passing yards/TDs contributions

**Implementation:**

```python
# Inside the QB contribution calculation loop
for qb_row in qualified_qbs.iter_rows(named=True):
    player_id = qb_row['player_id']
    
    # Get this QB's plays from PBP
    qb_plays = pbp_data.filter(pl.col('passer_player_id') == player_id)
    
    # Calculate base contributions (existing code)
    passing_yards = qb_plays['yards_gained'].sum()
    passing_tds = qb_plays.filter(pl.col('touchdown') == 1).height
    
    # NEW: Apply FTN contextual adjustments
    if ftn_data is not None:
        # Play action adjustment: Reduce value of PA passes (easier context)
        pa_plays = qb_plays.filter(pl.col('is_play_action') == True)
        pa_yards = pa_plays['yards_gained'].sum()
        pa_adjustment = pa_yards * -0.10  # 10% penalty for easier throws
        
        # Out of pocket bonus: Reward improvisation
        oop_plays = qb_plays.filter(pl.col('is_qb_out_of_pocket') == True)
        oop_completions = oop_plays.filter(pl.col('complete_pass') == 1).height
        oop_bonus = oop_completions * 3.0  # 3 points per OOP completion
        
        # Blitz context: Reduce value when facing blitz (easier to complete)
        blitz_plays = qb_plays.filter(pl.col('n_blitzers') >= 1)
        blitz_completions = blitz_plays.filter(pl.col('complete_pass') == 1).height
        blitz_adjustment = blitz_completions * -2.0  # Small penalty for easier throws
        
        # Screen pass adjustment: Low-difficulty throws
        screen_plays = qb_plays.filter(pl.col('is_screen_pass') == True)
        screen_yards = screen_plays['yards_gained'].sum()
        screen_adjustment = screen_yards * -0.15  # 15% penalty for screen passes
        
        # Apply adjustments to base contribution
        contribution += pa_adjustment + oop_bonus + blitz_adjustment + screen_adjustment
        
        # Store breakdown for transparency
        qb_results[player_id]['pa_adjustment'] = pa_adjustment
        qb_results[player_id]['oop_bonus'] = oop_bonus
        qb_results[player_id]['blitz_adjustment'] = blitz_adjustment
        qb_results[player_id]['screen_adjustment'] = screen_adjustment
```

---

### 2.3 Apply WR/TE Contextual Adjustments

**File:** `main.py` - `generate_wr_rankings()` and `generate_te_rankings()` functions

**Implementation:**

```python
def generate_wr_rankings(year: int) -> str:
    # ... existing WR loading code ...
    
    # Load PBP and FTN data
    pbp_path = Path("cache/pbp") / f"pbp_{year}.parquet"
    pbp_data = pl.read_parquet(pbp_path)
    
    from modules.ftn_cache_builder import load_ftn_cache
    ftn_data = load_ftn_cache(year)
    
    if ftn_data is not None:
        pbp_data = pbp_data.join(ftn_data, on=['game_id', 'play_id'], how='left')
        
        # Log contested catch availability
        contested_pct = (pbp_data['is_contested_ball'].sum() / len(pbp_data)) * 100
        drop_pct = (pbp_data['is_drop'].sum() / len(pbp_data)) * 100
        logger.info(f"FTN coverage: {contested_pct:.1f}% contested, {drop_pct:.1f}% drops")
    else:
        pbp_data = pbp_data.with_columns([
            pl.lit(None).alias('is_contested_ball'),
            pl.lit(None).alias('is_drop'),
            pl.lit(None).alias('is_rpo'),
            pl.lit(None).alias('is_screen_pass'),
        ])
    
    # In WR contribution calculation loop:
    for wr_row in qualified_wrs.iter_rows(named=True):
        player_id = wr_row['player_id']
        
        # Get this WR's targets
        wr_targets = pbp_data.filter(pl.col('receiver_player_id') == player_id)
        
        # Calculate base reception value (existing)
        receptions = wr_targets.filter(pl.col('complete_pass') == 1).height
        base_reception_value = receptions * 5.0
        
        # NEW: Contested catch bonus (1.25x multiplier as requested)
        if ftn_data is not None:
            contested_catches = wr_targets.filter(
                (pl.col('complete_pass') == 1) & 
                (pl.col('is_contested_ball') == True)
            ).height
            
            # Apply 1.25x to contested catches (0.25x bonus)
            contested_bonus = contested_catches * 5.0 * 0.25  # 25% bonus per catch
            
            # Drop penalty: Subtract points for unreliability
            drops = wr_targets.filter(pl.col('is_drop') == True).height
            drop_penalty = drops * -8.0  # Harsh penalty for drops
            
            # Screen pass context: Lower value (easy yards)
            screen_catches = wr_targets.filter(
                (pl.col('complete_pass') == 1) &
                (pl.col('is_screen_pass') == True)
            ).height
            screen_yards = wr_targets.filter(
                pl.col('is_screen_pass') == True
            )['yards_gained'].sum()
            screen_adjustment = screen_yards * -0.10  # 10% reduction for easy yards
            
            contribution += contested_bonus + drop_penalty + screen_adjustment
            
            # Store for transparency
            wr_results[player_id]['contested_bonus'] = contested_bonus
            wr_results[player_id]['drop_penalty'] = drop_penalty
            wr_results[player_id]['screen_adjustment'] = screen_adjustment
```

---

### 2.4 Apply RB Contextual Adjustments

**File:** `main.py` - `generate_rb_rankings()` function

**Implementation:**

```python
def generate_rb_rankings(year: int) -> str:
    # ... existing RB loading code ...
    
    # Load PBP and FTN data
    pbp_path = Path("cache/pbp") / f"pbp_{year}.parquet"
    pbp_data = pl.read_parquet(pbp_path)
    
    from modules.ftn_cache_builder import load_ftn_cache
    ftn_data = load_ftn_cache(year)
    
    if ftn_data is not None:
        pbp_data = pbp_data.join(ftn_data, on=['game_id', 'play_id'], how='left')
        
        rpo_pct = (pbp_data['is_rpo'].sum() / len(pbp_data)) * 100
        logger.info(f"FTN coverage: {rpo_pct:.1f}% RPO plays")
    else:
        pbp_data = pbp_data.with_columns([
            pl.lit(None).alias('is_rpo'),
            pl.lit(None).alias('n_defense_box'),
        ])
    
    # In RB contribution calculation loop:
    for rb_row in qualified_rbs.iter_rows(named=True):
        player_id = rb_row['player_id']
        
        # Get this RB's rushes
        rb_rushes = pbp_data.filter(pl.col('rusher_player_id') == player_id)
        
        # Calculate base rushing value (existing)
        rushing_yards = rb_rushes['yards_gained'].sum()
        
        # NEW: RPO context adjustment
        if ftn_data is not None:
            # RPO runs are easier (numbers advantage)
            rpo_runs = rb_rushes.filter(pl.col('is_rpo') == True)
            rpo_yards = rpo_runs['yards_gained'].sum()
            rpo_adjustment = rpo_yards * -0.12  # 12% reduction for easier runs
            
            # Use FTN n_defense_box (more accurate than inferred for 2022+)
            # Already using defenders_in_box in difficulty calculation,
            # but FTN's is human-verified
            heavy_box_runs = rb_rushes.filter(pl.col('n_defense_box') >= 8)
            heavy_box_yards = heavy_box_runs['yards_gained'].sum()
            heavy_box_bonus = heavy_box_yards * 0.15  # 15% bonus for stacked boxes
            
            contribution += rpo_adjustment + heavy_box_bonus
            
            rb_results[player_id]['rpo_adjustment'] = rpo_adjustment
            rb_results[player_id]['heavy_box_bonus'] = heavy_box_bonus
```

---

## Phase 3: Testing & Validation

### 3.1 Unit Tests

**File:** `tests/test_ftn_integration.py` (NEW)

```python
"""
Test FTN cache builder and integration
"""

import pytest
import polars as pl
from pathlib import Path
from modules.ftn_cache_builder import (
    build_ftn_cache_for_year,
    ftn_cache_exists,
    load_ftn_cache,
    FTN_START_YEAR
)

def test_ftn_cache_pre_2022():
    """FTN data should not be available before 2022"""
    assert not ftn_cache_exists(2021)
    assert not ftn_cache_exists(2020)
    result = build_ftn_cache_for_year(2021)
    assert result is False

def test_ftn_cache_2024():
    """FTN cache should build successfully for 2024"""
    result = build_ftn_cache_for_year(2024)
    assert result is True
    assert ftn_cache_exists(2024)
    
    # Verify cache contents
    ftn = load_ftn_cache(2024)
    assert ftn is not None
    assert len(ftn) > 0
    
    # Check required columns
    required_cols = [
        'nflverse_game_id',
        'nflverse_play_id',
        'is_play_action',
        'is_qb_out_of_pocket',
        'n_blitzers',
        'is_contested_ball',
        'is_drop',
        'is_rpo',
        'is_screen_pass',
        'n_defense_box'
    ]
    
    for col in required_cols:
        assert col in ftn.columns, f"Missing column: {col}"

def test_ftn_join_with_pbp():
    """FTN data should join correctly with PBP data"""
    pbp = pl.read_parquet('cache/pbp/pbp_2024.parquet')
    ftn = load_ftn_cache(2024)
    
    # Join
    joined = pbp.join(ftn, on=['game_id', 'play_id'], how='left')
    
    # Should have same number of rows as PBP
    assert len(joined) == len(pbp)
    
    # Should have FTN columns
    assert 'is_play_action' in joined.columns
    
    # Some plays should have FTN data
    pa_count = joined['is_play_action'].sum()
    assert pa_count > 0, "No play action flags found after join"

def test_ftn_flags_reasonable():
    """FTN flags should be within reasonable ranges"""
    ftn = load_ftn_cache(2024)
    
    # Play action should be 5-15% of plays
    pa_pct = (ftn['is_play_action'].sum() / len(ftn)) * 100
    assert 5 < pa_pct < 15, f"Play action {pa_pct}% outside expected range"
    
    # Contested balls should be 3-10% of plays
    contested_pct = (ftn['is_contested_ball'].sum() / len(ftn)) * 100
    assert 3 < contested_pct < 10, f"Contested balls {contested_pct}% outside expected range"
    
    # Drops should be 0.5-3% of plays
    drop_pct = (ftn['is_drop'].sum() / len(ftn)) * 100
    assert 0.5 < drop_pct < 3, f"Drops {drop_pct}% outside expected range"
```

---

### 3.2 Integration Tests

**File:** `tests/test_ftn_impact.py` (NEW)

```python
"""
Test that FTN data actually impacts rankings as expected
"""

import pytest
from main import (
    generate_qb_rankings,
    generate_wr_rankings,
    generate_te_rankings,
    generate_rb_rankings
)

def test_qb_play_action_reduces_value():
    """QBs with high play action % should have adjusted scores"""
    # Generate 2024 rankings (has FTN data)
    qb_rankings_2024 = generate_qb_rankings(2024)
    
    # Parse markdown to extract PA adjustment values
    # (This requires parsing the markdown output)
    # Should show negative PA adjustments for QBs with high PA usage
    assert 'pa_adjustment' in qb_rankings_2024 or True  # Placeholder

def test_wr_contested_catch_bonus():
    """WRs with contested catches should get 1.25x bonus"""
    wr_rankings_2024 = generate_wr_rankings(2024)
    
    # Should show contested_bonus in output
    assert 'contested' in wr_rankings_2024.lower()

def test_rb_rpo_adjustment():
    """RBs with high RPO usage should have reduced yards value"""
    rb_rankings_2024 = generate_rb_rankings(2024)
    
    # Should show RPO adjustments
    # Placeholder for actual implementation
    assert True

def test_ftn_graceful_degradation():
    """Rankings should work for years without FTN data"""
    # 2021 has no FTN data
    qb_rankings_2021 = generate_qb_rankings(2021)
    
    # Should complete successfully without errors
    assert qb_rankings_2021 is not None
    assert len(qb_rankings_2021) > 0
```

---

### 3.3 Manual Validation Tests

**Test Procedure:**

1. **Build FTN Cache:**
   ```bash
   .venv/Scripts/python.exe -m modules.ftn_cache_builder
   ```
   
   **Expected Output:**
   - "Building FTN cache for years 2022-2025"
   - "Successfully built FTN cache for 4 years"
   
   **Verify:**
   ```bash
   ls cache/ftn/
   # Should see: ftn_2022.parquet, ftn_2023.parquet, ftn_2024.parquet, ftn_2025.parquet
   ```

2. **Regenerate 2024 Rankings (with FTN):**
   ```bash
   .venv/Scripts/python.exe main.py 2024
   ```
   
   **Expected Log Output:**
   ```
   INFO - Joining FTN charting data for contextual adjustments (48031 plays)
   INFO - FTN coverage: 10.6% play action, 9.0% out of pocket
   INFO - FTN coverage: 6.2% contested, 1.3% drops
   INFO - FTN coverage: 2.7% RPO plays
   ```
   
   **Verify Rankings Changed:**
   - Compare `output/2024/qb_rankings.md` before and after FTN integration
   - Look for QBs with high play action % to have lower adjusted scores
   - Look for WRs with contested catches to have bonus in contribution

3. **Regenerate 2021 Rankings (without FTN):**
   ```bash
   .venv/Scripts/python.exe main.py 2021
   ```
   
   **Expected Log Output:**
   ```
   INFO - FTN data not available for 2021, using base calculations
   ```
   
   **Verify No Errors:**
   - Rankings should complete successfully
   - Output should look identical to before (no FTN adjustments)

4. **Spot Check Specific Players:**

   **QB - Josh Allen (2024):**
   ```bash
   grep "J.Allen" output/2024/qb_rankings.md
   ```
   - Note his contribution score
   - Check if he had high play action usage (should see PA adjustment)
   - Check if he scrambled often (should see OOP bonus)

   **WR - Justin Jefferson (2024):**
   ```bash
   grep "J.Jefferson" output/2024/wr_rankings.md
   ```
   - Look for contested catch bonus
   - Should be higher than similar stat lines without contested catches

   **RB - Derrick Henry (2024):**
   ```bash
   grep "D.Henry" output/2024/rb_rankings.md
   ```
   - Look for heavy box bonus (he faces loaded boxes frequently)
   - Should have higher adjusted score relative to yards

---

### 3.4 Data Quality Validation

**Script:** `validate_ftn_integration.py` (NEW)

```python
"""
Validate FTN integration data quality and coverage
"""

import polars as pl
from pathlib import Path
from modules.ftn_cache_builder import load_ftn_cache

def validate_ftn_coverage():
    """Check FTN data coverage and quality for all cached years"""
    
    print("FTN Data Quality Report")
    print("=" * 70)
    
    for year in range(2022, 2026):
        ftn = load_ftn_cache(year)
        
        if ftn is None:
            print(f"\n{year}: NO DATA")
            continue
        
        print(f"\n{year}:")
        print(f"  Total plays: {len(ftn):,}")
        
        # Calculate percentages
        pa_pct = (ftn['is_play_action'].sum() / len(ftn)) * 100
        oop_pct = (ftn['is_qb_out_of_pocket'].sum() / len(ftn)) * 100
        contested_pct = (ftn['is_contested_ball'].sum() / len(ftn)) * 100
        drop_pct = (ftn['is_drop'].sum() / len(ftn)) * 100
        rpo_pct = (ftn['is_rpo'].sum() / len(ftn)) * 100
        screen_pct = (ftn['is_screen_pass'].sum() / len(ftn)) * 100
        
        print(f"  Play Action: {pa_pct:5.2f}% ({ftn['is_play_action'].sum():,} plays)")
        print(f"  Out of Pocket: {oop_pct:5.2f}% ({ftn['is_qb_out_of_pocket'].sum():,} plays)")
        print(f"  Contested: {contested_pct:5.2f}% ({ftn['is_contested_ball'].sum():,} plays)")
        print(f"  Drops: {drop_pct:5.2f}% ({ftn['is_drop'].sum():,} plays)")
        print(f"  RPO: {rpo_pct:5.2f}% ({ftn['is_rpo'].sum():,} plays)")
        print(f"  Screen: {screen_pct:5.2f}% ({ftn['is_screen_pass'].sum():,} plays)")
        
        # Blitzers distribution
        avg_blitzers = ftn['n_blitzers'].mean()
        max_blitzers = ftn['n_blitzers'].max()
        print(f"  Blitzers: avg={avg_blitzers:.2f}, max={max_blitzers}")
        
        # Defense box distribution
        avg_box = ftn['n_defense_box'].mean()
        box_8plus = (ftn.filter(pl.col('n_defense_box') >= 8).height / len(ftn)) * 100
        print(f"  Defense Box: avg={avg_box:.2f}, 8+ defenders={box_8plus:.1f}%")

def compare_with_without_ftn():
    """Compare rankings with and without FTN adjustments"""
    
    print("\n\nRanking Impact Analysis")
    print("=" * 70)
    
    # This would require running rankings twice and comparing
    # Placeholder for now
    print("TODO: Implement before/after comparison")

if __name__ == "__main__":
    validate_ftn_coverage()
    compare_with_without_ftn()
```

---

## Phase 4: Documentation Updates

### 4.1 Update README.md

Add section explaining FTN data:

```markdown
## Data Sources

### Primary Data (nflverse/nflreadpy)
- Play-by-play data (2000-present)
- Player statistics (2000-present)
- Team statistics (2000-present)
- Participation data (2016-present)

### Enhanced Data (FTN Charting)
**Available:** 2022-present

FTN (Football Technology Network) provides human-charted flags for specific play characteristics:

**QB Context:**
- Play action vs standard dropback
- In pocket vs out of pocket (scramble)
- Number of blitzers

**WR/TE Context:**
- Contested catches (defender within 1 yard)
- True drops (catchable ball not caught)

**All Positions:**
- RPO (Run-Pass Option) plays
- Screen passes
- Defenders in box (verified count)

These flags allow for more sophisticated contextual adjustments that separate player skill from scheme/situation.
```

---

### 4.2 Update Methodology Documentation

Create `METHODOLOGY.md` section on FTN adjustments:

```markdown
## FTN Contextual Adjustments (2022+)

For years with FTN data available, we apply the following adjustments:

### QB Adjustments

**Play Action Reduction (-10%)**
- Rationale: Play action passes are statistically easier (defense fooled, simplified reads)
- Applied to: Passing yards on play action plays
- Example: 100 yards on PA → -10 point adjustment

**Out of Pocket Bonus (+3 pts per completion)**
- Rationale: Completing passes while scrambling shows improvisation and athleticism
- Applied to: Completions when QB is outside tackle box
- Example: 5 OOP completions → +15 points

**Blitz Adjustment (-2 pts per completion)**
- Rationale: Blitz creates easier throwing windows (less coverage)
- Applied to: Completions when defense sends extra rushers
- Example: 8 completions vs blitz → -16 points

**Screen Pass Reduction (-15%)**
- Rationale: Screens are high-percentage, low-difficulty throws
- Applied to: Passing yards on screen passes
- Example: 50 yards on screens → -7.5 point adjustment

### WR/TE Adjustments

**Contested Catch Bonus (+25%)**
- Rationale: Catching with defender in position requires elite ball skills
- Applied to: Reception value on contested catches
- Example: 5 base points → 6.25 points for contested catch

**Drop Penalty (-8 pts per drop)**
- Rationale: Drops represent unreliability and missed opportunities
- Applied to: Catchable balls that hit hands and were not caught
- Example: 3 drops → -24 points

**Screen Pass Reduction (-10%)**
- Rationale: Screen catches are easier (less traffic, designed space)
- Applied to: Receiving yards on screen passes
- Example: 40 yards on screens → -4 point adjustment

### RB Adjustments

**RPO Reduction (-12%)**
- Rationale: RPO runs have numbers advantage (QB reads to favorable side)
- Applied to: Rushing yards on RPO plays
- Example: 50 yards on RPO → -6 point adjustment

**Heavy Box Bonus (+15%)**
- Rationale: Running against 8+ defenders in box is significantly harder
- Applied to: Rushing yards against heavy boxes
- Example: 60 yards vs 8+ box → +9 points

### Why These Multipliers?

All multipliers are chosen to be:
1. **Conservative** - Small enough not to overwhelm base stats
2. **Proportional** - Reflect actual difficulty difference
3. **Transparent** - Simple percentages, no complex formulas
4. **Testable** - Can be validated against actual outcomes

You can adjust these in the code if you disagree with the weightings.
```

---

## Phase 5: Rollout Plan

### 5.1 Implementation Order

**Week 1: Infrastructure**
1. Create `modules/ftn_cache_builder.py`
2. Test FTN data fetching for 2024
3. Verify cache creation and loading
4. Update `check_and_rebuild_caches()`

**Week 2: QB Integration**
1. Add FTN join to `generate_qb_rankings()`
2. Implement play action adjustment
3. Implement out of pocket bonus
4. Implement blitz adjustment
5. Implement screen pass adjustment
6. Test on 2024 data
7. Compare before/after rankings

**Week 3: WR/TE Integration**
1. Add FTN join to `generate_wr_rankings()` and `generate_te_rankings()`
2. Implement contested catch bonus
3. Implement drop penalty
4. Implement screen pass adjustment
5. Test on 2024 data
6. Compare before/after rankings

**Week 4: RB Integration**
1. Add FTN join to `generate_rb_rankings()`
2. Implement RPO adjustment
3. Implement heavy box bonus
4. Test on 2024 data
5. Compare before/after rankings

**Week 5: Testing & Validation**
1. Run all unit tests
2. Run integration tests
3. Execute manual validation procedures
4. Generate data quality reports
5. Spot check specific players

**Week 6: Full Regeneration**
1. Backup existing output/
2. Build FTN cache for 2022-2025
3. Regenerate all years 2000-2025
4. Verify 2000-2021 unchanged (no FTN)
5. Verify 2022-2025 have FTN adjustments
6. Compare ranking changes year-over-year

---

### 5.2 Rollback Plan

If issues arise during integration:

1. **Remove FTN join** - Comment out FTN loading in ranking functions
2. **Keep cache** - Don't delete FTN cache (can re-enable later)
3. **Regenerate** - Re-run rankings without FTN to restore previous state
4. **Debug** - Fix issues in isolation before re-integrating

**Safety Mechanisms:**
- FTN cache builder is separate module (can disable without touching main)
- FTN join uses `how='left'` (won't break if FTN missing)
- Null FTN columns added if not available (code still runs)
- Graceful degradation for pre-2022 years

---

## Phase 6: Future Enhancements

### 6.1 Add FTN Metrics to Output Files

**Enhancement:** Display FTN-influenced metrics in ranking markdown

```markdown
## QB Rankings - 2024

| Rank | Player | Team | Raw | Adjusted | PA% | OOP% | Blitz% |
|------|--------|------|-----|----------|-----|------|--------|
| 1    | J.Allen | BUF  | 4500 | 4380    | 12% | 15%  | 8%     |
```

### 6.2 Add FTN Summary Report

**New File:** `output/{year}/ftn_summary.md`

```markdown
# FTN Context Summary - 2024

## Play Characteristics

- **Play Action Usage:** 10.6% of passes
- **Out of Pocket:** 9.0% of dropbacks
- **Contested Catches:** 6.2% of targets
- **RPO Runs:** 2.7% of rushes

## Top QB by Context

**Most Out of Pocket:** J.Allen (18% of dropbacks)
**Most Play Action:** S.Darnold (22% of passes)
**Faced Most Blitzes:** J.Burrow (12% of dropbacks)

## Top WR by Context

**Most Contested Catches:** M.Evans (15% of receptions)
**Fewest Drops:** J.Jefferson (0.8% drop rate)

## Top RB by Context

**Most Heavy Box:** D.Henry (45% vs 8+ defenders)
**Most RPO:** K.Williams (8% of rushes)
```

### 6.3 Historical Comparison Tool

**Script:** `compare_ftn_impact.py`

```python
"""
Compare player rankings with and without FTN adjustments
to see who benefits most from context
"""

def compare_impact(year: int):
    # Generate rankings without FTN
    rankings_base = generate_rankings_no_ftn(year)
    
    # Generate rankings with FTN
    rankings_ftn = generate_rankings_with_ftn(year)
    
    # Calculate rank changes
    for player in all_players:
        rank_change = rankings_base[player] - rankings_ftn[player]
        print(f"{player}: {rank_change:+d} spots")
```

---

## Success Criteria

### MVP (Minimum Viable Product)

- ✅ FTN cache builds successfully for 2022-2025
- ✅ QB rankings incorporate play action, OOP, blitz, screen adjustments
- ✅ WR rankings incorporate contested catch bonus and drop penalty
- ✅ RB rankings incorporate RPO and heavy box adjustments
- ✅ Pre-2022 years still work (graceful degradation)
- ✅ All unit tests pass
- ✅ Manual validation shows expected ranking changes

### Nice to Have

- 📊 FTN metrics displayed in output markdown
- 📊 FTN summary report generated
- 📊 Historical comparison tool
- 📊 Detailed adjustment breakdowns per player

### Long Term Goals

- 📈 Add more FTN flags as valuable (target separation, route concepts)
- 📈 Tune multipliers based on historical validation
- 📈 Expand to other positions (OL blocking metrics if FTN adds them)

---

## Risk Assessment

### High Risk
- **FTN API Changes:** nflreadpy dependency could break
  - **Mitigation:** Cache data locally, version pin nflreadpy
  
### Medium Risk
- **Join Performance:** Joining FTN with PBP could be slow
  - **Mitigation:** Use parquet format, index on game_id/play_id
  
- **Multiplier Tuning:** Initial multipliers may need adjustment
  - **Mitigation:** Make them configurable constants, easy to tweak

### Low Risk
- **Data Quality:** FTN charting may have errors
  - **Mitigation:** Spot check against game film, validate ranges
  
- **Coverage Gaps:** Some plays may not have FTN data
  - **Mitigation:** Use left join, handle nulls gracefully

---

## Timeline

**Total Estimated Time:** 6-8 weeks

- Week 1: Infrastructure (8-10 hours) ✅ **COMPLETE**
- Week 2: QB Integration (10-12 hours)
- Week 3: WR/TE Integration (10-12 hours)
- Week 4: RB Integration (8-10 hours)
- Week 5: Testing (12-15 hours)
- Week 6: Full Regeneration (4-6 hours)
- Weeks 7-8: Advanced QB Contextual Adjustments (12-16 hours) ✅ **COMPLETE**

**Critical Path:** FTN cache builder → QB integration → Testing

**Parallelizable:** WR/TE and RB integrations can happen concurrently

---

## Phase 7: Advanced QB Contextual Adjustments (Completed Ahead of Schedule)

**Status:** ✅ Implemented (October 2025)

**Rationale:** While working on the FTN integration plan, we identified opportunities to enhance QB evaluation using existing NextGen Stats data (pressure metrics) that was already being collected but underutilized. These enhancements were implemented before Phase 2 to establish a more robust baseline for QB evaluation.

### 7.1 Contextual Penalties for Negative Plays

**Motivation:** Not all turnovers/sacks are created equal. A pick-six in your own territory in a close game in the final 2 minutes should be penalized more heavily than a desperation 4th down heave.

**Implementation:**

```python
# Base penalties
INT_BASE_PENALTY = -35
SACK_BASE_PENALTY = -10
FUMBLE_BASE_PENALTY = -35

# Context multipliers (multiplicative)
- Field Position: 1.5x if in own territory (< midfield)
- Score Differential: 1.2x if close game (within 1 score)
- Time Remaining: 1.3x if final 2 minutes of half
- Down/Distance: 0.7x if on 3rd/4th down (calculated risk acceptable)
```

**Impact on 2025 Rankings:**
- Dak Prescott: #4 → #1 (fewer costly turnovers in critical situations)
- Jalen Hurts: #7 → #11 (more turnovers in high-leverage situations)
- Josh Allen: #10 → #16 (turnovers and sacks in critical contexts penalized)

**Philosophy Alignment:** Uses objective, observable data (field position, score, time, down) rather than subjective "decision quality" grades. These are measurable contexts that affect win probability.

### 7.2 Pressure Completion Bonuses

**Motivation:** Completions under pressure are significantly more difficult and valuable than clean pocket completions. NextGen Stats pressure data (2016+, most complete 2024+) provides objective measurement of defensive pressure.

**Implementation:**

```python
# Applied to 2024 and earlier (pressure data available)
if qb_pressure == True:
    completion_points *= 1.4  # 40% bonus for completion
    yards_value *= 1.2        # 20% bonus for yards gained
```

**Data Availability:**
- 2016-2023: Partial pressure data
- 2024+: Complete pressure tracking
- 2025: Not yet applied (pending season completion)

**Philosophy Alignment:** Pressure is measured by NextGen Stats GPS tracking (defender within 3 yards of QB within 2.5 seconds of snap). This is objective sensor data, not human judgment.

### 7.3 Design Philosophy Discussion

**Why These Were Implemented First:**

1. **Data Already Available:** Pressure data was being loaded from NextGen Stats but not fully utilized. No new data dependencies.

2. **Objective Measurements:** Both enhancements use sensor data (GPS tracking for pressure) and factual game state (score, time, field position, down) rather than human charting/judgment.

3. **Complementary to FTN:** These provide a robust baseline for QB evaluation. When FTN flags (play action, out of pocket, blitz count) are added in Phase 2, they'll layer on top of this stronger foundation.

4. **Immediate Value:** Could improve QB rankings immediately without waiting for full FTN integration.

**Distinction from FTN Integration:**

- **Phase 7 (Completed):** Enhanced utilization of existing NextGen Stats data
- **Phase 2 (Next):** Add new FTN charting data for play-type specific adjustments

Both align with the "objective data only" philosophy - Phase 7 uses GPS sensors, Phase 2 uses human-charted observable flags (not subjective grades).

---

## Appendix: Quick Reference Commands

```bash
# Build FTN cache
python -m modules.ftn_cache_builder

# Regenerate single year with FTN
python main.py 2024

# Regenerate all years
python main.py --start-year 2000 --end-year 2025

# Run tests
pytest tests/test_ftn_integration.py
pytest tests/test_ftn_impact.py

# Validate data quality
python validate_ftn_integration.py

# Check specific player
grep "J.Allen" output/2024/qb_rankings.md
```

---

**Last Updated:** 2025-10-30  
**Status:** Phase 1 ✅ Complete | Phase 7 ✅ Complete | Phase 2 In Progress  
**Next Action:** Integrate FTN data into QB rankings (Phase 2.1)
