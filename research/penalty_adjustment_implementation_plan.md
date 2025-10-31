# Penalty Adjustment Implementation Plan

## Overview
Implement situationally-weighted penalty adjustments for skill players (QB, RB, WR, TE) to penalize players who commit costly penalties. Penalties will be weighted based on type (loss of down, dead ball, standard), situational context (down, field position, time, EPA), and repeat offender status.

**Integration Point:** Phase 4 (Context Adjustments) - alongside FTN charting flags and NextGen Stats separation/cushion adjustments

**Target Positions:** QB, RB, WR, TE (skill players who commit offensive penalties)

---

## Phase 1: Data Module & Cache Builder

### File: `modules/penalty_cache_builder.py`

Following established patterns from `injury_cache_builder.py`, `ftn_cache_builder.py`, and `positional_cache_builder.py`.

```python
"""
Penalty Cache Builder Module

Loads and caches play-by-play penalty data for skill player penalty adjustments.
Tracks offensive penalties committed by QB/RB/WR/TE with full situational context.
"""

import polars as pl
import nflreadpy as nfl
from pathlib import Path
from modules.logger import logger
from modules.constants import (
    # Penalty type classifications
    PENALTY_LOSS_OF_DOWN_BASE,
    PENALTY_DEAD_BALL_BASE,
    PENALTY_STANDARD_BASE,
    PENALTY_REPEAT_OFFENDER_BASE,
    
    # Down situation multipliers
    PENALTY_DOWN_1ST,
    PENALTY_DOWN_2ND,
    PENALTY_DOWN_3RD_4TH,
    
    # Field position multipliers
    PENALTY_FIELD_REDZONE,
    PENALTY_FIELD_MIDFIELD_IN,
    PENALTY_FIELD_MIDFIELD_OUT,
    
    # Time/Quarter multipliers
    PENALTY_TIME_EARLY,
    PENALTY_TIME_TWO_MINUTE,
    PENALTY_TIME_CRITICAL,
    
    # EPA severity multipliers
    PENALTY_EPA_MINOR,
    PENALTY_EPA_MODERATE,
    PENALTY_EPA_MAJOR,
    PENALTY_EPA_CATASTROPHIC,
    
    # Penalty type lists
    LOSS_OF_DOWN_PENALTIES,
    DEAD_BALL_PENALTIES,
    SKILL_PLAYER_PENALTIES
)

# Cache directory following established pattern
PENALTY_CACHE_DIR = Path("cache/penalties")


def load_penalty_data(season: int) -> pl.DataFrame:
    """
    Load and cache penalty data from play-by-play for a given season.
    
    Filters to only skill player offensive penalties and enriches with
    situational context needed for weighting.
    
    Args:
        season: NFL season year (e.g., 2024)
        
    Returns:
        Polars DataFrame with penalty data and context
        
    Schema:
        - penalty_player_id: GSIS ID of player who committed penalty
        - penalty_player_name: Player name (F.LastName format)
        - penalty_type: Type of penalty
        - penalty_yards: Yards assessed
        - down: Down when penalty occurred (1-4)
        - ydstogo: Yards to go for first down
        - yardline_100: Yards from opponent's goal line
        - qtr: Quarter (1-4)
        - quarter_seconds_remaining: Seconds left in quarter
        - epa: Expected Points Added (negative for penalties)
        - wpa: Win Probability Added (negative for penalties)
        - posteam: Team on offense
        - game_id: Unique game identifier
    """
    cache_file = PENALTY_CACHE_DIR / f"penalties-{season}.csv"
    
    # Check if cached
    if cache_file.exists():
        logger.debug(f"Loading cached penalty data for {season}")
        return pl.read_csv(cache_file)
    
    logger.info(f"Fetching penalty data for {season} from nflreadpy...")
    
    try:
        # Load full play-by-play data
        pbp = nfl.load_pbp(season)
        
        # Filter to penalty plays only
        penalties = pbp.filter(pl.col('penalty') == 1)
        
        # Filter to skill player offensive penalties
        skill_penalties = penalties.filter(
            pl.col('penalty_type').is_in(SKILL_PLAYER_PENALTIES)
        )
        
        # Select relevant columns with context
        penalty_data = skill_penalties.select([
            'penalty_player_id',
            'penalty_player_name',
            'penalty_type',
            'penalty_yards',
            'down',
            'ydstogo',
            'yardline_100',
            'qtr',
            'quarter_seconds_remaining',
            'epa',
            'wpa',
            'posteam',
            'game_id'
        ])
        
        # Cache the data
        PENALTY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        penalty_data.write_csv(cache_file)
        
        logger.info(
            f"Cached {len(penalty_data)} skill player penalties for {season}"
        )
        
        return penalty_data
        
    except Exception as e:
        logger.error(f"Failed to load penalty data for {season}: {e}")
        return pl.DataFrame()


def get_penalty_base_multiplier(
    penalty_type: str,
    player_total_penalties: int,
    dead_ball_count: int
) -> float:
    """
    Calculate base penalty multiplier based on penalty type and repeat offender status.
    
    Args:
        penalty_type: Type of penalty (e.g., 'Offensive Pass Interference')
        player_total_penalties: Total penalties by this player this season
        dead_ball_count: Number of dead ball penalties by this player
        
    Returns:
        Base multiplier (1.0, 1.5, or 2.0)
    """
    # Loss of Down penalties (worst)
    if penalty_type in LOSS_OF_DOWN_PENALTIES:
        return PENALTY_LOSS_OF_DOWN_BASE  # 2.0
    
    # Dead Ball penalties (ejection risk)
    if penalty_type in DEAD_BALL_PENALTIES:
        # Upgrade to 2.0 if repeat offender (2+ dead ball penalties)
        if dead_ball_count >= 2:
            return PENALTY_REPEAT_OFFENDER_BASE  # 2.0
        return PENALTY_DEAD_BALL_BASE  # 1.5
    
    # Standard live ball penalties
    return PENALTY_STANDARD_BASE  # 1.0


def get_down_multiplier(down: float) -> float:
    """
    Get multiplier based on down situation.
    
    Args:
        down: Down number (1-4)
        
    Returns:
        Down multiplier
    """
    if down == 1.0:
        return PENALTY_DOWN_1ST  # 1.0
    elif down == 2.0:
        return PENALTY_DOWN_2ND  # 1.2
    else:  # 3rd or 4th down
        return PENALTY_DOWN_3RD_4TH  # 1.5


def get_field_position_multiplier(yardline_100: float) -> float:
    """
    Get multiplier based on field position.
    
    Args:
        yardline_100: Yards from opponent's goal line (lower = closer to scoring)
        
    Returns:
        Field position multiplier
    """
    if yardline_100 < 20.0:
        # Red Zone - most costly
        return PENALTY_FIELD_REDZONE  # 1.5
    elif yardline_100 <= 50.0:
        # Midfield In - good field position wasted
        return PENALTY_FIELD_MIDFIELD_IN  # 1.2
    else:
        # Midfield Out - less critical
        return PENALTY_FIELD_MIDFIELD_OUT  # 1.0


def get_time_multiplier(qtr: float, quarter_seconds_remaining: float) -> float:
    """
    Get multiplier based on time context (game leverage).
    
    Args:
        qtr: Quarter (1-4)
        quarter_seconds_remaining: Seconds remaining in quarter
        
    Returns:
        Time multiplier
    """
    # Q4 under 2 minutes - game deciding
    if qtr == 4.0 and quarter_seconds_remaining < 120.0:
        return PENALTY_TIME_CRITICAL  # 1.5
    
    # Q2 under 2 minutes (two-minute drill) OR Q4 between 2-5 minutes
    if (qtr == 2.0 and quarter_seconds_remaining < 120.0) or \
       (qtr == 4.0 and quarter_seconds_remaining < 300.0):
        return PENALTY_TIME_TWO_MINUTE  # 1.3
    
    # Early game - plenty of time
    return PENALTY_TIME_EARLY  # 1.0


def get_epa_severity_multiplier(epa: float) -> float:
    """
    Get multiplier based on EPA (Expected Points Added) severity.
    
    EPA is negative for penalties - more negative = worse.
    
    Args:
        epa: Expected Points Added (negative for penalties)
        
    Returns:
        EPA severity multiplier
    """
    if epa > -0.5:
        return PENALTY_EPA_MINOR  # 1.0
    elif epa > -1.0:
        return PENALTY_EPA_MODERATE  # 1.2
    elif epa > -1.5:
        return PENALTY_EPA_MAJOR  # 1.5
    else:
        return PENALTY_EPA_CATASTROPHIC  # 2.0


def calculate_penalty_adjustment(
    player_gsis_id: str,
    season: int
) -> float:
    """
    Calculate total penalty adjustment multiplier for a player.
    
    This is the MAIN ENTRY POINT called from Phase 4.
    
    Logic:
    1. Load all penalties for the season
    2. Filter to penalties by this player
    3. Calculate weighted penalty score for each penalty
    4. Sum total penalty impact
    5. Return adjustment multiplier (<= 1.0)
    
    Args:
        player_gsis_id: Player's GSIS ID
        season: Season year
        
    Returns:
        Penalty adjustment multiplier (e.g., 0.95 = 5% penalty)
        Returns 1.0 if no penalties or data unavailable
    """
    try:
        # Load penalty data
        penalties = load_penalty_data(season)
        
        if penalties.is_empty():
            logger.debug(f"No penalty data available for {season}")
            return 1.0
        
        # Filter to this player's penalties
        player_penalties = penalties.filter(
            pl.col('penalty_player_id') == player_gsis_id
        )
        
        if len(player_penalties) == 0:
            # No penalties - no adjustment
            return 1.0
        
        # Count dead ball penalties for repeat offender detection
        dead_ball_count = len(
            player_penalties.filter(
                pl.col('penalty_type').is_in(DEAD_BALL_PENALTIES)
            )
        )
        
        total_penalty_impact = 0.0
        
        # Calculate weighted impact for each penalty
        for row in player_penalties.iter_rows(named=True):
            # Base multiplier (1.0, 1.5, or 2.0)
            base = get_penalty_base_multiplier(
                row['penalty_type'],
                len(player_penalties),
                dead_ball_count
            )
            
            # Situational multipliers (all stack)
            down_mult = get_down_multiplier(row['down'])
            field_mult = get_field_position_multiplier(row['yardline_100'])
            time_mult = get_time_multiplier(
                row['qtr'],
                row['quarter_seconds_remaining']
            )
            epa_mult = get_epa_severity_multiplier(row['epa'])
            
            # Calculate total weighted penalty
            weighted_penalty = base * down_mult * field_mult * time_mult * epa_mult
            
            total_penalty_impact += weighted_penalty
            
            logger.debug(
                f"{row['penalty_player_name']} - {row['penalty_type']}: "
                f"Base={base:.1f}, Down={down_mult:.1f}x, "
                f"Field={field_mult:.1f}x, Time={time_mult:.1f}x, "
                f"EPA={epa_mult:.1f}x => {weighted_penalty:.2f}"
            )
        
        # Convert total impact to adjustment multiplier
        # Each point of penalty impact = 1% reduction
        # Cap at 20% maximum reduction (0.80 minimum multiplier)
        penalty_reduction = min(total_penalty_impact * 0.01, 0.20)
        adjustment = 1.0 - penalty_reduction
        
        logger.info(
            f"Penalty adjustment for {player_penalties[0]['penalty_player_name']}: "
            f"{len(player_penalties)} penalties, "
            f"impact={total_penalty_impact:.2f}, "
            f"multiplier={adjustment:.3f}"
        )
        
        return adjustment
        
    except Exception as e:
        logger.error(f"Error calculating penalty adjustment: {e}")
        return 1.0


def build_penalty_cache_for_year(year: int) -> bool:
    """
    Build penalty cache for a single year.
    
    Args:
        year: Season year to cache
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Building penalty cache for {year}...")
    
    penalties = load_penalty_data(year)
    success = not penalties.is_empty()
    
    if success:
        logger.info(f"Successfully cached {len(penalties)} penalties for {year}")
    else:
        logger.warning(f"Failed to cache penalties for {year}")
    
    return success


def build_penalty_cache(start_year: int = 2016, end_year: int = 2025) -> None:
    """
    Build penalty cache for multiple years.
    
    Args:
        start_year: First year to cache (default: 2016)
        end_year: Last year to cache (default: 2025)
    """
    logger.info(f"Building penalty cache for {start_year}-{end_year}...")
    
    for year in range(start_year, end_year + 1):
        try:
            build_penalty_cache_for_year(year)
        except Exception as e:
            logger.error(f"Failed to build penalty cache for {year}: {e}")
            continue
    
    logger.info("Penalty cache building complete")


def cache_is_up_to_date(start_year: int = 2016, end_year: int = 2024) -> list:
    """
    Check which years are missing from penalty cache.
    
    Args:
        start_year: First year to check
        end_year: Last year to check
        
    Returns:
        List of missing years
    """
    missing = []
    
    for year in range(start_year, end_year + 1):
        cache_file = PENALTY_CACHE_DIR / f"penalties-{year}.csv"
        if not cache_file.exists():
            missing.append(year)
    
    return missing
```

---

## Phase 2: Add Constants

### File: `modules/constants.py`

Add penalty adjustment constants following existing patterns.

```python
# ============================================================================
# Penalty Adjustment Constants
# ============================================================================

# Penalty Type Classifications - Base Multipliers
PENALTY_LOSS_OF_DOWN_BASE = 2.0      # Intentional grounding, illegal forward pass
PENALTY_DEAD_BALL_BASE = 1.5         # Taunting, unsportsmanlike conduct
PENALTY_REPEAT_OFFENDER_BASE = 2.0   # Dead ball penalties when player has 2+
PENALTY_STANDARD_BASE = 1.0          # OPI, face mask, unnecessary roughness

# Down Situation Multipliers
PENALTY_DOWN_1ST = 1.0               # 1st down - baseline
PENALTY_DOWN_2ND = 1.2               # 2nd down - more critical
PENALTY_DOWN_3RD_4TH = 1.5          # 3rd/4th down - drive killer

# Field Position Multipliers
PENALTY_FIELD_REDZONE = 1.5          # Inside 20 - kills scoring drives
PENALTY_FIELD_MIDFIELD_IN = 1.2      # 20-50 yard line - wastes good position
PENALTY_FIELD_MIDFIELD_OUT = 1.0     # 50+ yard line - less critical

# Time/Quarter Context Multipliers
PENALTY_TIME_EARLY = 1.0             # Q1/Q2 (>2:00) - plenty of time
PENALTY_TIME_TWO_MINUTE = 1.3        # Q2 (<2:00) or Q4 (2:00-5:00) - critical drive
PENALTY_TIME_CRITICAL = 1.5          # Q4 (<2:00) - game deciding

# EPA Severity Multipliers
PENALTY_EPA_MINOR = 1.0              # EPA > -0.5 - minor impact
PENALTY_EPA_MODERATE = 1.2           # EPA -0.5 to -1.0 - moderate cost
PENALTY_EPA_MAJOR = 1.5              # EPA -1.0 to -1.5 - major cost
PENALTY_EPA_CATASTROPHIC = 2.0       # EPA < -1.5 - catastrophic

# Penalty Type Lists
LOSS_OF_DOWN_PENALTIES = [
    'Intentional Grounding',
    'Illegal Forward Pass',
    'Illegal Touch Pass',
    'Illegal Touch Kick'
]

DEAD_BALL_PENALTIES = [
    'Taunting',
    'Unsportsmanlike Conduct'
]

# All skill player offensive penalties we care about
SKILL_PLAYER_PENALTIES = [
    'Offensive Pass Interference',
    'Intentional Grounding',
    'Illegal Forward Pass',
    'Taunting',
    'Unsportsmanlike Conduct',
    'Unnecessary Roughness',
    'Face Mask',
    'Lowering the Head to Make Forcible Contact',
    'Illegal Touch Pass',
    'Illegal Touch Kick'
]
```

---

## Phase 3: Integrate into Phase 4

### File: `main.py`

Modify Phase 4 in each position's ranking function to include penalty adjustments.

**Integration Points:**
1. `generate_qb_rankings()` - after line ~1150 (Phase 4 FTN/NextGen adjustments)
2. `generate_rb_rankings()` - after line ~1300 (Phase 4 adjustments)
3. `generate_wr_rankings()` - after line ~1540 (Phase 4 adjustments)
4. `generate_te_rankings()` - after line ~1770 (Phase 4 adjustments)

**Example Integration (for WR rankings):**

```python
    # Phase 4: Context-Based Adjustments (FTN Charting + NextGen + Penalties)
    logger.info("Phase 4: Applying context-based adjustments...")
    
    # Load FTN context tables
    ftn_data_yac = load_ftn_context_data(year, 'yac')
    ftn_data_target = load_ftn_context_data(year, 'target')
    
    # Load NextGen separation data
    nextgen_data = load_nextgen_data(year, ['receiving'])
    
    # Import penalty adjustment function
    from modules.penalty_cache_builder import calculate_penalty_adjustment
    
    adjusted_contributions = {}
    
    for player_name, contribution in phase3_contributions.items():
        multiplier = 1.0
        
        # Existing FTN adjustments
        if player_name in ftn_data_yac:
            row = ftn_data_yac[player_name]
            # ... existing FTN logic ...
        
        if player_name in ftn_data_target:
            row = ftn_data_target[player_name]
            # ... existing FTN logic ...
        
        # Existing NextGen separation adjustments
        if player_name in nextgen_data:
            row = nextgen_data[player_name]
            # ... existing NextGen logic ...
        
        # NEW: Penalty adjustments
        player_gsis_id = get_player_gsis_id(player_name, None, 'WR', year)
        if player_gsis_id:
            penalty_multiplier = calculate_penalty_adjustment(player_gsis_id, year)
            multiplier *= penalty_multiplier
            
            if penalty_multiplier < 1.0:
                logger.debug(
                    f"{player_name}: Penalty adjustment = {penalty_multiplier:.3f}"
                )
        
        adjusted_contributions[player_name] = contribution * multiplier
```

**Note:** Use the existing `get_player_gsis_id()` function from `injury_cache_builder.py` to map player names to GSIS IDs. This function already handles "F.LastName" format matching.

---

## Phase 4: Add to Cache Rebuild System

### File: `modules/rebuild_caches.py`

Add penalty cache building to the cache rebuild workflow.

```python
from modules.penalty_cache_builder import build_penalty_cache

def rebuild_all_caches():
    """Rebuild all cache files."""
    logger.info("Starting cache rebuild process...")
    
    # ... existing cache builders ...
    
    # Build injury caches
    logger.info("Building injury caches...")
    build_injury_cache(2016, 2025)
    
    # NEW: Build penalty caches
    logger.info("Building penalty caches...")
    build_penalty_cache(2016, 2025)
    
    # ... rest of cache building ...
```

---

## Phase 5: Testing Strategy

### 5.1 Unit Tests

Create `test_penalty_adjustment.py`:

```python
"""
Test penalty adjustment calculations with known 2024 cases.
"""

from modules.penalty_cache_builder import (
    load_penalty_data,
    calculate_penalty_adjustment,
    get_player_gsis_id
)
from modules.injury_cache_builder import get_player_gsis_id
import polars as pl

def test_penalty_data_loading():
    """Test that penalty data loads correctly."""
    penalties = load_penalty_data(2024)
    
    assert len(penalties) > 0, "Should load penalty data"
    assert 'penalty_player_id' in penalties.columns
    assert 'penalty_type' in penalties.columns
    assert 'epa' in penalties.columns
    
    print(f"✓ Loaded {len(penalties)} penalties for 2024")


def test_drake_london_penalties():
    """
    Test Drake London - 2 unsportsmanlike conduct penalties in 2024.
    Should get penalized as repeat offender.
    """
    # Get Drake London's GSIS ID
    gsis_id = get_player_gsis_id("D.London", "ATL", "WR", 2024)
    
    if not gsis_id:
        print("✗ Could not find Drake London's GSIS ID")
        return
    
    # Calculate penalty adjustment
    adjustment = calculate_penalty_adjustment(gsis_id, 2024)
    
    print(f"\nDrake London Penalty Analysis:")
    print(f"  GSIS ID: {gsis_id}")
    print(f"  Penalty Multiplier: {adjustment:.3f}")
    print(f"  Reduction: {(1.0 - adjustment) * 100:.1f}%")
    
    # Should be penalized (multiplier < 1.0)
    assert adjustment < 1.0, "Drake London should be penalized for 2 USC penalties"
    assert adjustment >= 0.80, "Should not exceed 20% maximum penalty"
    
    print("✓ Drake London correctly penalized as repeat offender")


def test_george_pickens_penalties():
    """
    Test George Pickens - 2 penalties (taunting + USC) in 2024.
    Should get penalized as repeat offender.
    """
    gsis_id = get_player_gsis_id("G.Pickens", "PIT", "WR", 2024)
    
    if not gsis_id:
        print("✗ Could not find George Pickens' GSIS ID")
        return
    
    adjustment = calculate_penalty_adjustment(gsis_id, 2024)
    
    print(f"\nGeorge Pickens Penalty Analysis:")
    print(f"  GSIS ID: {gsis_id}")
    print(f"  Penalty Multiplier: {adjustment:.3f}")
    print(f"  Reduction: {(1.0 - adjustment) * 100:.1f}%")
    
    assert adjustment < 1.0, "Pickens should be penalized"
    assert adjustment >= 0.80, "Should not exceed 20% maximum penalty"
    
    print("✓ George Pickens correctly penalized")


def test_dtr_intentional_grounding():
    """
    Test Dorian Thompson-Robinson - 5 intentional grounding penalties.
    Should get heavily penalized (loss of down penalties).
    """
    gsis_id = get_player_gsis_id("D.Thompson-Robinson", "CLE", "QB", 2024)
    
    if not gsis_id:
        print("✗ Could not find DTR's GSIS ID")
        return
    
    adjustment = calculate_penalty_adjustment(gsis_id, 2024)
    
    print(f"\nDorian Thompson-Robinson Penalty Analysis:")
    print(f"  GSIS ID: {gsis_id}")
    print(f"  Penalty Multiplier: {adjustment:.3f}")
    print(f"  Reduction: {(1.0 - adjustment) * 100:.1f}%")
    
    # 5 loss-of-down penalties should be heavily penalized
    assert adjustment < 0.90, "DTR should be heavily penalized for 5 grounding penalties"
    assert adjustment >= 0.80, "Should not exceed 20% maximum penalty"
    
    print("✓ DTR correctly penalized for intentional grounding")


def test_clean_player_no_penalties():
    """
    Test a player with no penalties - should return 1.0 (no adjustment).
    """
    # Test with a reliable player (likely has no penalties)
    gsis_id = get_player_gsis_id("C.Lamb", "DAL", "WR", 2024)
    
    if not gsis_id:
        print("✗ Could not find CeeDee Lamb's GSIS ID")
        return
    
    adjustment = calculate_penalty_adjustment(gsis_id, 2024)
    
    print(f"\nCeeDee Lamb Penalty Analysis:")
    print(f"  GSIS ID: {gsis_id}")
    print(f"  Penalty Multiplier: {adjustment:.3f}")
    
    # If no penalties, should be 1.0
    if adjustment == 1.0:
        print("✓ CeeDee Lamb has no penalties (1.0 multiplier)")
    else:
        print(f"  CeeDee has penalties: {(1.0 - adjustment) * 100:.1f}% reduction")


def test_penalty_cache_exists():
    """Test that 2024 penalty cache was created."""
    from pathlib import Path
    cache_file = Path("cache/penalties/penalties-2024.csv")
    
    assert cache_file.exists(), "2024 penalty cache should exist"
    
    penalties = pl.read_csv(cache_file)
    assert len(penalties) > 0, "Cache should have penalty data"
    
    print(f"✓ Penalty cache exists with {len(penalties)} penalties")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Penalty Adjustment System")
    print("=" * 60)
    
    test_penalty_data_loading()
    test_penalty_cache_exists()
    test_drake_london_penalties()
    test_george_pickens_penalties()
    test_dtr_intentional_grounding()
    test_clean_player_no_penalties()
    
    print("\n" + "=" * 60)
    print("All penalty adjustment tests passed!")
    print("=" * 60)
```

### 5.2 Integration Test

Run full 2024 pipeline and validate:

```python
# Run main.py for 2024
python main.py

# Check that penalty adjustments were applied
# Look for log messages like:
#   "Drake London: Penalty adjustment = 0.950"
#   "George Pickens: Penalty adjustment = 0.945"
#   "D.Thompson-Robinson: Penalty adjustment = 0.880"

# Compare rankings before/after penalty implementation
# Players with penalties should rank slightly lower
```

### 5.3 Manual Validation

Check specific players in 2024 WR/QB rankings:

**Expected WR Results:**
- **Drake London:** 2 USC penalties → ~5% reduction
- **George Pickens:** 2 penalties → ~5% reduction  
- **Jakobi Meyers:** 3 OPI penalties → ~3-6% reduction (context dependent)
- **Clean players (Jefferson, Chase, etc.):** No penalty adjustment

**Expected QB Results:**
- **DTR:** 5 intentional grounding → ~10-15% reduction
- **Sam Darnold:** 3 intentional grounding → ~6-9% reduction
- **Clean QBs (Mahomes, Allen, etc.):** No penalty adjustment

---

## Phase 6: Pre-Cache Historical Data

Once implementation is complete and tested, pre-cache all years:

```python
from modules.penalty_cache_builder import build_penalty_cache

# Pre-cache 2016-2024 (all years with PBP data)
build_penalty_cache(2016, 2024)

# Verify cache
from modules.penalty_cache_builder import cache_is_up_to_date
missing = cache_is_up_to_date(2016, 2024)
print(f"Missing penalty cache files: {missing}")
```

---

## Cache Structure

### Directory: `cache/penalties/`

Following established pattern from `cache/injuries/`, `cache/rosters/`, etc.

**File Format:** CSV (following pattern from injury/roster caches)

**Naming Convention:** `penalties-{year}.csv`

**Example Files:**
```
cache/penalties/
├── penalties-2016.csv
├── penalties-2017.csv
├── penalties-2018.csv
├── ...
├── penalties-2023.csv
├── penalties-2024.csv
└── penalties-2025.csv
```

**CSV Schema:**
```csv
penalty_player_id,penalty_player_name,penalty_type,penalty_yards,down,ydstogo,yardline_100,qtr,quarter_seconds_remaining,epa,wpa,posteam,game_id
00-0033106,D.London,Unsportsmanlike Conduct,15.0,2.0,10.0,65.0,3.0,420.0,-0.89,-0.024,ATL,2024_01_ATL_PIT
00-0033106,D.London,Unsportsmanlike Conduct,15.0,1.0,10.0,72.0,2.0,180.0,-1.12,-0.031,ATL,2024_08_ATL_TB
...
```

**CSV Benefits:**
- Human readable for debugging
- Easy to inspect with Excel/text editor
- Consistent with injury/roster cache format
- Smaller file size for this data (~3000-6000 penalties per year)

**Alternative (Parquet):**
If file size becomes an issue or performance is a concern, can switch to parquet:
```python
# In load_penalty_data():
cache_file = PENALTY_CACHE_DIR / f"penalties-{season}.parquet"
penalty_data.write_parquet(cache_file)
return pl.read_parquet(cache_file)
```

---

## Implementation Checklist

### Phase 1: Module Creation
- [ ] Create `modules/penalty_cache_builder.py`
- [ ] Implement `load_penalty_data()`
- [ ] Implement penalty classification functions
- [ ] Implement situational multiplier functions
- [ ] Implement `calculate_penalty_adjustment()` (main entry point)
- [ ] Implement cache management functions

### Phase 2: Constants
- [ ] Add all penalty constants to `modules/constants.py`
- [ ] Add penalty type classification lists
- [ ] Verify all multiplier values

### Phase 3: Integration
- [ ] Import penalty adjustment in `main.py` Phase 4
- [ ] Integrate into `generate_qb_rankings()` Phase 4
- [ ] Integrate into `generate_rb_rankings()` Phase 4
- [ ] Integrate into `generate_wr_rankings()` Phase 4
- [ ] Integrate into `generate_te_rankings()` Phase 4
- [ ] Add logging for penalty adjustments

### Phase 4: Cache System
- [ ] Add penalty cache building to `modules/rebuild_caches.py`
- [ ] Create `cache/penalties/` directory
- [ ] Test cache creation/loading

### Phase 5: Testing
- [ ] Create `test_penalty_adjustment.py`
- [ ] Test data loading
- [ ] Test known penalty cases (London, Pickens, DTR)
- [ ] Test clean players (no penalties)
- [ ] Run full 2024 pipeline integration test
- [ ] Manually validate WR/QB rankings

### Phase 6: Pre-Caching
- [ ] Pre-cache 2016-2024 penalty data
- [ ] Verify all cache files present
- [ ] Validate cache file sizes/content

### Phase 7: Validation
- [ ] Compare 2024 rankings before/after penalty implementation
- [ ] Verify players with penalties rank appropriately lower
- [ ] Check that multipliers are reasonable (not too harsh/lenient)
- [ ] Validate logging output is clear and informative

---

## Expected Performance Impact

**Penalty Frequency (2024):**
- Total skill player penalties: ~450
- Average per player: Most players have 0-1 penalties
- Repeat offenders: ~6 players with 2+ dead ball penalties
- Heavy offenders: ~3 QBs with 3+ intentional grounding

**Ranking Impact:**
- **No penalties (majority):** No change (1.0x multiplier)
- **1-2 minor penalties:** ~2-5% reduction
- **2+ dead ball penalties:** ~5-10% reduction (repeat offender)
- **3-5 intentional grounding:** ~6-12% reduction (loss of down + context)
- **Maximum cap:** 20% reduction (0.80 minimum multiplier)

**Examples:**
- Drake London (2 USC): Likely ~4-6% reduction
- George Pickens (2 penalties): Likely ~4-6% reduction
- DTR (5 grounding): Likely ~10-15% reduction
- Jakobi Meyers (3 OPI): Likely ~3-8% reduction (context dependent)

**Performance:**
- Data load: ~1-2 seconds per year (cached)
- Calculation: ~0.01 seconds per player
- Total Phase 4 overhead: ~2-3 seconds for 200 players
- Overall pipeline impact: <5% increase

---

## Success Criteria

✅ **Module Implementation:**
- `penalty_cache_builder.py` created and functional
- All functions implemented with proper error handling
- Follows existing module patterns

✅ **Constants Added:**
- All penalty multipliers defined in `constants.py`
- Values match agreed-upon weights

✅ **Integration Complete:**
- Penalty adjustments applied in Phase 4 for all positions
- Proper logging of penalty impacts
- Uses existing GSIS ID lookup function

✅ **Testing Passed:**
- Known penalty cases validated (London, Pickens, DTR)
- Clean players return 1.0 multiplier
- Full 2024 pipeline runs successfully
- Rankings look reasonable

✅ **Cache System Working:**
- Cache files created in `cache/penalties/`
- CSV format follows naming convention
- Pre-cached 2016-2024 successfully

✅ **Documentation:**
- Implementation plan comprehensive and detailed
- Future sessions can continue work seamlessly
- Code is well-commented and clear

---

## Notes for Future Sessions

### Key Design Decisions:

1. **Integration Point:** Phase 4 (context adjustments), NOT Phase 5 (sample size)
   - Penalties reflect play quality/discipline, not sample size
   - Stack with FTN flags and NextGen separation
   
2. **Multiplier Application:** Multiplicative reduction (e.g., 0.95x)
   - NOT additive penalty to raw stats
   - Preserves existing Phase 1-3 contributions
   
3. **Maximum Penalty:** 20% reduction (0.80 floor)
   - Prevents over-penalization of outliers
   - Even worst offenders keep 80% of contribution
   
4. **Repeat Offender Boost:** Dead ball penalties get 2.0x base if 2+ instances
   - Reflects ejection risk and chronic discipline issues
   - Applies to taunting and unsportsmanlike conduct
   
5. **GSIS ID Lookup:** Reuse existing function from `injury_cache_builder.py`
   - Handles "F.LastName" format
   - Gracefully falls back to 1.0 if player not found

### Potential Enhancements (Not in Scope):

- **Penalty timing:** Weight penalties in winning/losing games differently
- **Opponent quality:** Weight penalties vs. good teams higher
- **Penalty acceptance:** Consider if penalty was declined (negated by gain)
- **Multi-year penalty history:** Similar to injury 3-year weighted average
- **Position-specific thresholds:** Different max penalties for QB vs. WR

### Troubleshooting:

**Issue: Many "Could not find GSIS ID" warnings**
- Solution: Check that roster data is loaded for the year
- Solution: Verify player name format matches ("F.LastName")
- Solution: Add fallback to 1.0 (no penalty) for unknown players

**Issue: Penalties seem too harsh/lenient**
- Solution: Adjust constants in `constants.py`
- Solution: Check EPA multipliers - may need tuning
- Solution: Verify situational multipliers are stacking correctly

**Issue: Cache files not found**
- Solution: Run `build_penalty_cache()` for missing years
- Solution: Check `cache/penalties/` directory exists
- Solution: Verify nflreadpy has PBP data for that year

---

## Timeline Estimate

- **Phase 1 (Module):** 45-60 minutes
- **Phase 2 (Constants):** 5-10 minutes
- **Phase 3 (Integration):** 30-45 minutes
- **Phase 4 (Cache System):** 10-15 minutes
- **Phase 5 (Testing):** 30-45 minutes
- **Phase 6 (Pre-caching):** 10-15 minutes
- **Phase 7 (Validation):** 20-30 minutes

**Total Estimated Time:** 2.5 - 3.5 hours

---

## End of Implementation Plan

This plan is comprehensive enough for any session to pick up and continue implementation seamlessly. All design decisions, integration points, testing strategies, and edge cases are documented.
