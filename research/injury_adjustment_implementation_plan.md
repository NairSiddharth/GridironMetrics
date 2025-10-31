# Injury-Adjusted Sample Size Implementation Plan

## Executive Summary

Implement a 3-year weighted injury history system that adjusts Phase 5 sample size dampening to distinguish between:
- **Durable players** (minimal penalty for missed games)
- **Injury-prone players** (full penalty for missed games)
- **One-off injury cases** (moderate penalty)

This ensures our talent evaluation properly accounts for whether missed games reflect injury risk vs. bad luck.

---

## Architecture Overview

### Integration Point: Phase 5 Sample Size Dampening

**Current Phase 5 Logic** (in `main.py`):
```python
# Around line 800-850 in apply_phase5_adjustments()
sample_size_factor = (games_played / max_games) ** 0.4
```

**New Logic**:
```python
# Calculate injury-adjusted effective games
effective_games = calculate_injury_adjusted_games(player_id, season, games_played, max_games)
sample_size_factor = (effective_games / max_games) ** 0.4
```

### File Structure

```
modules/
├── injury_cache_builder.py     [NEW] - Injury data loader and cache builder
└── constants.py                [MODIFY] - Add injury-related constants

cache/
├── injuries/                   [NEW] - Cached injury data
│   ├── injuries-2016.csv
│   ├── injuries-2017.csv
│   └── ... (through 2025)
└── rosters/                    [NEW] - Cached roster data
    ├── rosters-2016.csv
    ├── rosters-2017.csv
    └── ... (through 2025)

main.py                         [MODIFY] - Integrate into Phase 5
```

---

## Implementation Steps

### Step 1: Create Data Loader Module

**File**: `modules/injury_cache_builder.py`

**Purpose**: Load and cache injury/roster data from nflreadpy

```python
"""
injury_cache_builder.py

Builds and maintains a cache of NFL injury reports and weekly roster status
from nflreadpy. Used for injury adjustment system in Phase 5 sample size dampening.

This module loads injury reports and weekly roster data to calculate
injury-adjusted games played, distinguishing chronic injury risk from one-off injuries.
"""

import polars as pl
import nflreadpy as nfl
from pathlib import Path
from modules.logger import get_logger
from modules.constants import (
    CACHE_DIR,
    INJURY_RELIABLE_THRESHOLD,
    INJURY_MODERATE_THRESHOLD,
    INJURY_PRONE_THRESHOLD,
    INJURY_YEAR1_WEIGHT,
    INJURY_YEAR2_WEIGHT,
    INJURY_YEAR3_WEIGHT,
    RECURRING_INJURY_MULTIPLIER,
    ONEOFF_INJURY_MULTIPLIER,
)

logger = get_logger(__name__)

# Cache directories (following existing cache structure pattern)
CACHE_DIR = Path(CACHE_DIR)
INJURY_CACHE = CACHE_DIR / "injuries"
ROSTER_CACHE = CACHE_DIR / "rosters"

# Ensure cache directories exist
INJURY_CACHE.mkdir(parents=True, exist_ok=True)
ROSTER_CACHE.mkdir(parents=True, exist_ok=True)


def load_injury_data(season: int) -> pl.DataFrame:
    """
    Load injury report data for a specific season.
    
    Uses nflreadpy.load_injuries() and caches locally to avoid repeated API calls.
    
    Args:
        season: NFL season year (e.g., 2024)
        
    Returns:
        DataFrame with columns:
        - season (int)
        - week (int)
        - gsis_id (str) - Player identifier
        - full_name (str)
        - team (str)
        - position (str)
        - report_status (str) - 'Out', 'Questionable', 'Doubtful', etc.
        - report_primary_injury (str) - Body part/injury type
        - report_secondary_injury (str)
    """
    cache_file = INJURY_CACHE / f"injuries-{season}.csv"
    
    # Try to load from cache
    if cache_file.exists():
        logger.debug(f"Loading injury data from cache: {cache_file}")
        return pl.read_csv(cache_file)
    
    # Load from nflreadpy
    logger.info(f"Fetching injury data for {season} from nflreadpy...")
    try:
        injuries = nfl.load_injuries(seasons=season)
        
        # Save to cache
        injuries.write_csv(cache_file)
        logger.info(f"Cached injury data to {cache_file}")
        
        return injuries
    except Exception as e:
        logger.error(f"Failed to load injury data for {season}: {e}")
        return pl.DataFrame()


def load_roster_data(season: int) -> pl.DataFrame:
    """
    Load weekly roster status data for a specific season.
    
    Uses nflreadpy.load_rosters_weekly() and caches locally.
    
    Args:
        season: NFL season year (e.g., 2024)
        
    Returns:
        DataFrame with columns:
        - season (int)
        - week (int)
        - gsis_id (str) - Player identifier
        - full_name (str)
        - team (str)
        - position (str)
        - status (str) - 'ACT', 'INA', 'RES', etc.
        - jersey_number (int)
    """
    cache_file = ROSTER_CACHE / f"rosters-{season}.csv"
    
    # Try to load from cache
    if cache_file.exists():
        logger.debug(f"Loading roster data from cache: {cache_file}")
        return pl.read_csv(cache_file)
    
    # Load from nflreadpy
    logger.info(f"Fetching roster data for {season} from nflreadpy...")
    try:
        rosters = nfl.load_rosters_weekly(seasons=season)
        
        # Save to cache
        rosters.write_csv(cache_file)
        logger.info(f"Cached roster data to {cache_file}")
        
        return rosters
    except Exception as e:
        logger.error(f"Failed to load roster data for {season}: {e}")
        return pl.DataFrame()


def count_games_missed_due_to_injury(
    player_gsis_id: str,
    season: int,
    max_games: int = 17
) -> dict:
    """
    Count games a player missed specifically due to injury (not healthy scratches).
    
    Logic:
    - Load weekly roster status (ACT/INA/RES)
    - Load injury reports (Out/Questionable/etc.)
    - Cross-reference to distinguish injury from coaching decisions
    
    Args:
        player_gsis_id: Player's GSIS ID
        season: Season to check
        max_games: Max regular season games (17 for modern NFL)
        
    Returns:
        dict with:
        - games_played: Count of ACT status weeks
        - injury_missed: Count of weeks with (INA or RES) AND report_status='Out'
        - other_inactive: Count of weeks with INA but no injury report
        - injury_types: List of primary injuries that caused missed games
    """
    # Load data
    rosters = load_roster_data(season)
    injuries = load_injury_data(season)
    
    # Filter to this player (regular season only, weeks 1-17)
    player_rosters = rosters.filter(
        (pl.col('gsis_id') == player_gsis_id) &
        (pl.col('week') <= max_games)
    )
    
    player_injuries = injuries.filter(
        (pl.col('gsis_id') == player_gsis_id) &
        (pl.col('week') <= max_games)
    )
    
    if len(player_rosters) == 0:
        logger.warning(f"No roster data found for player {player_gsis_id} in {season}")
        return {
            'games_played': 0,
            'injury_missed': 0,
            'other_inactive': 0,
            'injury_types': []
        }
    
    # Join roster and injury data
    combined = player_rosters.join(
        player_injuries.select(['week', 'report_status', 'report_primary_injury']),
        on='week',
        how='left'
    )
    
    # Count games played
    games_played = len(combined.filter(pl.col('status') == 'ACT'))
    
    # Count injury-related misses (INA or RES with injury report)
    injury_missed_df = combined.filter(
        (pl.col('status').is_in(['INA', 'RES'])) &
        (pl.col('report_status') == 'Out')
    )
    injury_missed = len(injury_missed_df)
    
    # Count non-injury inactives
    other_inactive = len(combined.filter(
        (pl.col('status') == 'INA') &
        (pl.col('report_status').is_null() | (pl.col('report_status') != 'Out'))
    ))
    
    # Extract injury types
    injury_types = injury_missed_df.select('report_primary_injury').to_series().to_list()
    injury_types = [i for i in injury_types if i is not None]
    
    return {
        'games_played': games_played,
        'injury_missed': injury_missed,
        'other_inactive': other_inactive,
        'injury_types': injury_types
    }


def classify_injury_pattern(injury_history: list[dict]) -> tuple[str, float]:
    """
    Classify a player's injury pattern based on 3-year history.
    
    Applies injury type multipliers:
    - Recurring injuries (same body part multiple years): 1.25x
    - One-off traumatic injuries (broken bones, ACL): 0.5x
    
    Args:
        injury_history: List of dicts from count_games_missed_due_to_injury()
                       Ordered [Year 1 (current), Year 2 (last year), Year 3 (2 years ago)]
        
    Returns:
        (classification, multiplier) tuple:
        - classification: 'reliable', 'moderate', 'injury-prone'
        - multiplier: Adjustment factor for effective games calculation
    """
    if not injury_history or len(injury_history) == 0:
        return ('reliable', 1.0)
    
    # Collect all injury types across years
    all_injuries = []
    for year_data in injury_history:
        all_injuries.extend(year_data.get('injury_types', []))
    
    # Check for recurring injuries (same body part appears multiple times)
    injury_counts = {}
    for injury in all_injuries:
        if injury:
            injury_lower = injury.lower()
            injury_counts[injury_lower] = injury_counts.get(injury_lower, 0) + 1
    
    has_recurring = any(count >= 2 for count in injury_counts.values())
    
    # Check for one-off traumatic injuries
    traumatic_keywords = ['fracture', 'broken', 'acl', 'mcl', 'torn ligament']
    has_traumatic = any(
        any(keyword in injury.lower() for keyword in traumatic_keywords)
        for injury in all_injuries
        if injury
    )
    
    # Calculate weighted average of games missed
    weights = [INJURY_YEAR1_WEIGHT, INJURY_YEAR2_WEIGHT, INJURY_YEAR3_WEIGHT]
    expected_missed = 0.0
    
    for i, year_data in enumerate(injury_history):
        if i >= len(weights):
            break
        
        missed = year_data.get('injury_missed', 0)
        
        # Apply injury type multipliers
        if has_recurring:
            missed *= RECURRING_INJURY_MULTIPLIER
        elif has_traumatic:
            missed *= ONEOFF_INJURY_MULTIPLIER
        
        expected_missed += missed * weights[i]
    
    # Classify based on expected missed games
    if expected_missed < INJURY_RELIABLE_THRESHOLD:
        classification = 'reliable'
        multiplier = 0.25  # Minimal penalty
    elif expected_missed < INJURY_MODERATE_THRESHOLD:
        classification = 'moderate'
        multiplier = 0.50  # Moderate penalty
    elif expected_missed < INJURY_PRONE_THRESHOLD:
        classification = 'elevated'
        multiplier = 0.75  # Elevated penalty
    else:
        classification = 'injury-prone'
        multiplier = 1.00  # Full penalty
    
    logger.debug(
        f"Injury classification: {classification} "
        f"(expected {expected_missed:.1f} missed games, "
        f"recurring={has_recurring}, traumatic={has_traumatic})"
    )
    
    return (classification, multiplier)


def calculate_injury_adjusted_games(
    player_gsis_id: str,
    current_season: int,
    games_played: int,
    max_games: int = 17
) -> float:
    """
    Calculate injury-adjusted effective games for Phase 5 dampening.
    
    This is the main entry point called from Phase 5.
    
    Logic:
    1. Load 3-year injury history (current season and 2 prior)
    2. Classify player as reliable/moderate/injury-prone
    3. Adjust effective games based on classification
    
    Args:
        player_gsis_id: Player's GSIS ID
        current_season: Season being analyzed
        games_played: Actual games played this season
        max_games: Max regular season games (17)
        
    Returns:
        effective_games: Adjusted game count for sample size dampening
        
    Example:
        - Player played 10 games
        - Classified as "reliable" (minimal injury history)
        - effective_games = 10 + (17-10) * 0.75 = 15.25
        - Uses 15.25 for dampening instead of 10
    """
    # Gather 3-year history
    injury_history = []
    for year_offset in range(3):
        season = current_season - year_offset
        
        if season < 2016:  # Don't go before our data range
            break
        
        year_data = count_games_missed_due_to_injury(
            player_gsis_id, season, max_games
        )
        injury_history.append(year_data)
    
    # Handle rookies/young players (less than 3 years)
    if len(injury_history) == 0:
        # No history - assume reliable but give no benefit
        return float(games_played)
    
    # Classify injury pattern
    classification, penalty_multiplier = classify_injury_pattern(injury_history)
    
    # Calculate effective games
    games_missed = max_games - games_played
    
    # Apply graduated benefit based on classification
    # benefit_multiplier: how much of the missed games we "forgive"
    benefit_multiplier = 1.0 - penalty_multiplier
    
    effective_games = games_played + (games_missed * benefit_multiplier)
    
    logger.debug(
        f"Player {player_gsis_id} in {current_season}: "
        f"played {games_played}/{max_games}, "
        f"classified as '{classification}', "
        f"effective games = {effective_games:.2f}"
    )
    
    return effective_games


def get_player_gsis_id(player_name: str, team: str, position: str, season: int) -> str:
    """
    Get GSIS ID for a player using roster data.
    
    Args:
        player_name: Full player name (e.g., "Christian McCaffrey")
        team: Team abbreviation (e.g., "SF")
        position: Position code (e.g., "RB")
        season: Season year
        
    Returns:
        GSIS ID string, or empty string if not found
    """
    rosters = load_roster_data(season)
    
    # Try exact match first
    match = rosters.filter(
        (pl.col('full_name') == player_name) &
        (pl.col('team') == team) &
        (pl.col('position') == position)
    )
    
    if len(match) > 0:
        return match.select('gsis_id').item(0, 0)
    
    # Try fuzzy match on name only
    match = rosters.filter(
        pl.col('full_name').str.contains(player_name.split()[-1])  # Last name
    )
    
    if len(match) == 1:
        logger.warning(f"Fuzzy matched {player_name} to {match['full_name'].item(0)}")
        return match.select('gsis_id').item(0, 0)
    
    logger.error(f"Could not find GSIS ID for {player_name} ({team} {position})")
    return ""
```

---

### Step 2: Add Constants

**File**: `modules/constants.py`

**Add to existing constants**:

```python
# ============================================================================
# INJURY ADJUSTMENT CONSTANTS (Phase 5)
# ============================================================================

# 3-Year Weighted Average Weights
INJURY_YEAR1_WEIGHT = 0.50  # Current season
INJURY_YEAR2_WEIGHT = 0.30  # Last season
INJURY_YEAR3_WEIGHT = 0.20  # 2 seasons ago

# Expected Games Missed Thresholds
INJURY_RELIABLE_THRESHOLD = 3.0    # < 3 expected missed = reliable
INJURY_MODERATE_THRESHOLD = 5.0    # 3-5 expected missed = moderate risk
INJURY_PRONE_THRESHOLD = 7.0       # >= 7 expected missed = injury-prone

# Injury Type Multipliers
RECURRING_INJURY_MULTIPLIER = 1.25  # Recurring soft tissue injuries
ONEOFF_INJURY_MULTIPLIER = 0.50     # Traumatic one-off injuries (broken bones, ACL)

# Penalty Multipliers (how much to penalize missed games)
# benefit_multiplier = 1.0 - penalty_multiplier
INJURY_PENALTY_RELIABLE = 0.25   # Forgive 75% of missed games
INJURY_PENALTY_MODERATE = 0.50   # Forgive 50% of missed games
INJURY_PENALTY_ELEVATED = 0.75   # Forgive 25% of missed games
INJURY_PENALTY_PRONE = 1.00      # Forgive 0% of missed games
```

---

### Step 3: Modify Phase 5 Function

**File**: `main.py`

**Location**: `apply_phase5_adjustments()` function (around line 800-900)

**Current Code** (approximately):
```python
def apply_phase5_adjustments(contributions: pl.DataFrame) -> pl.DataFrame:
    """Apply Phase 5: Talent context and sample size adjustments"""
    
    logger.info("Applying Phase 5: Talent context dampening and sample size adjustments")
    
    # ... existing code for talent context ...
    
    # Sample size adjustment
    adjusted = adjusted.with_columns([
        ((pl.col('games') / 17) ** 0.4).alias('sample_size_factor')
    ])
    
    # ... rest of function ...
```

**New Code**:
```python
def apply_phase5_adjustments(contributions: pl.DataFrame) -> pl.DataFrame:
    """Apply Phase 5: Talent context and sample size adjustments"""
    
    logger.info("Applying Phase 5: Talent context dampening and sample size adjustments")
    
    # Import injury cache builder functions
    from modules.injury_cache_builder import calculate_injury_adjusted_games, get_player_gsis_id
    
    # ... existing code for talent context ...
    
    # Get current season from data
    current_season = contributions.select('season').unique().item(0, 0)
    
    # Calculate injury-adjusted effective games for each player
    logger.info("Calculating injury-adjusted effective games...")
    
    # Add GSIS ID if not present
    if 'gsis_id' not in contributions.columns:
        contributions = contributions.with_columns([
            pl.struct(['player', 'team', 'position']).map_elements(
                lambda x: get_player_gsis_id(
                    x['player'], x['team'], x['position'], current_season
                ),
                return_dtype=pl.Utf8
            ).alias('gsis_id')
        ])
    
    # Calculate effective games for each player
    effective_games_list = []
    for row in contributions.iter_rows(named=True):
        gsis_id = row.get('gsis_id', '')
        games_played = row.get('games', 0)
        
        if gsis_id:
            effective_games = calculate_injury_adjusted_games(
                gsis_id, current_season, games_played, max_games=17
            )
        else:
            # Fallback to actual games if no GSIS ID found
            effective_games = float(games_played)
        
        effective_games_list.append(effective_games)
    
    contributions = contributions.with_columns([
        pl.Series('effective_games', effective_games_list)
    ])
    
    # Sample size adjustment using effective games
    adjusted = adjusted.with_columns([
        ((pl.col('effective_games') / 17) ** 0.4).alias('sample_size_factor')
    ])
    
    logger.info("Injury adjustment complete")
    
    # ... rest of function ...
```

---

### Step 4: Update Position-Specific Rankings

**Files**: `main.py` - Functions to modify:
- `generate_qb_rankings()`
- `generate_rb_rankings()`
- `generate_wr_rankings()`
- `generate_te_rankings()`

**Change**: These functions already call `apply_phase5_adjustments()`, so no changes needed if Phase 5 is implemented correctly.

**Verification**: Ensure GSIS IDs are available in the contribution data before Phase 5.

---

### Step 5: Add Cache Building Support

**Purpose**: Support batch cache building for all years

**File**: `modules/injury_cache_builder.py` (add to end of file)

```python
def build_injury_cache_for_year(year: int) -> bool:
    """
    Build injury cache for a single year.
    
    Args:
        year: Season year to cache
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Building injury cache for {year}...")
    
    # Load data (this will cache it automatically via load_injury_data)
    injuries = load_injury_data(year)
    rosters = load_roster_data(year)
    
    success = (not injuries.is_empty()) and (not rosters.is_empty())
    
    if success:
        logger.info(f"Successfully cached injury data for {year}")
    else:
        logger.warning(f"No data cached for {year}")
    
    return success


def build_injury_cache(start_year: int, end_year: int):
    """
    Build injury and roster caches for a range of years.
    
    Args:
        start_year: First year to cache
        end_year: Last year to cache
    """
    logger.info(f"Building injury cache for {start_year}-{end_year}...")
    
    for year in range(start_year, end_year + 1):
        build_injury_cache_for_year(year)
    
    logger.info("Injury cache build complete")


def cache_is_up_to_date(start_year: int, end_year: int) -> list:
    """
    Check if injury cache is complete for the given year range.
    
    Args:
        start_year: First year to check
        end_year: Last year to check
        
    Returns:
        List of (year, data_type) tuples for missing cache files
    """
    missing = []
    
    for year in range(start_year, end_year + 1):
        injury_file = INJURY_CACHE / f"injuries-{year}.csv"
        roster_file = ROSTER_CACHE / f"rosters-{year}.csv"
        
        if not injury_file.exists():
            missing.append((year, 'injuries'))
        if not roster_file.exists():
            missing.append((year, 'rosters'))
    
    return missing
```

**File**: `modules/rebuild_caches.py` (add to existing file)

```python
# Add to imports at top
from modules.injury_cache_builder import build_injury_cache

# Add to rebuild_all_caches() function
def rebuild_all_caches(start_year, end_year):
    """Rebuild all data caches"""
    # ... existing cache rebuilds ...
    
    # Add injury cache rebuild
    logger.info("Rebuilding injury cache...")
    build_injury_cache(start_year, end_year)
```

---

### Step 6: Add Caching Strategy

**Purpose**: Avoid repeated API calls to nflreadpy

**Implementation**:

1. **On first run**: Download injury/roster data for all years (2016-2025)
2. **Cache locally**: Store as CSV in `cache/injuries/` and `cache/rosters/`
3. **On subsequent runs**: Load from cache unless explicitly invalidated
4. **Cache invalidation**: Provide flag or delete cache files to force refresh

**Add helper script**: `scripts/refresh_injury_cache.py`

```python
"""
Refresh injury and roster data cache.

Run this script to download fresh data from nflreadpy for all seasons.
"""

import nflreadpy as nfl
from pathlib import Path
import polars as pl

CACHE_DIR = Path("cache")
INJURY_CACHE = CACHE_DIR / "injuries"
ROSTER_CACHE = CACHE_DIR / "rosters"

INJURY_CACHE.mkdir(parents=True, exist_ok=True)
ROSTER_CACHE.mkdir(parents=True, exist_ok=True)

START_YEAR = 2016
END_YEAR = 2025

print(f"Refreshing injury and roster cache for {START_YEAR}-{END_YEAR}...")

for year in range(START_YEAR, END_YEAR + 1):
    print(f"\nYear {year}:")
    
    # Injuries
    try:
        print(f"  Loading injury data...")
        injuries = nfl.load_injuries(seasons=year)
        injury_file = INJURY_CACHE / f"injuries-{year}.csv"
        injuries.write_csv(injury_file)
        print(f"  ✓ Cached {len(injuries)} injury records to {injury_file}")
    except Exception as e:
        print(f"  ✗ Failed to load injury data: {e}")
    
    # Rosters
    try:
        print(f"  Loading roster data...")
        rosters = nfl.load_rosters_weekly(seasons=year)
        roster_file = ROSTER_CACHE / f"rosters-{year}.csv"
        rosters.write_csv(roster_file)
        print(f"  ✓ Cached {len(rosters)} roster records to {roster_file}")
    except Exception as e:
        print(f"  ✗ Failed to load roster data: {e}")

print("\n✓ Cache refresh complete!")
```

---

## Testing Strategy

### Test 1: Unit Test - Single Player

**File**: `tests/test_injury_cache_builder.py` (create new)

```python
import pytest
from modules.injury_cache_builder import (
    count_games_missed_due_to_injury,
    classify_injury_pattern,
    calculate_injury_adjusted_games
)

def test_cmc_injury_history():
    """Test Christian McCaffrey's known injury history"""
    
    # CMC GSIS ID: Need to look this up
    gsis_id = "00-0033908"  # Placeholder
    
    # 2024: Missed 13 games (Achilles)
    result_2024 = count_games_missed_due_to_injury(gsis_id, 2024)
    assert result_2024['injury_missed'] == 13
    assert 'Achilles' in str(result_2024['injury_types'])
    
    # 2023: Played 16-17 games
    result_2023 = count_games_missed_due_to_injury(gsis_id, 2023)
    assert result_2023['games_played'] >= 16
    
    # 2022: Mostly healthy
    result_2022 = count_games_missed_due_to_injury(gsis_id, 2022)
    assert result_2022['games_played'] >= 15

def test_injury_classification():
    """Test injury pattern classification logic"""
    
    # Reliable player (minimal missed games)
    reliable_history = [
        {'injury_missed': 0, 'injury_types': []},
        {'injury_missed': 1, 'injury_types': ['Ankle']},
        {'injury_missed': 0, 'injury_types': []}
    ]
    classification, multiplier = classify_injury_pattern(reliable_history)
    assert classification == 'reliable'
    assert multiplier == 0.25
    
    # Injury-prone player
    prone_history = [
        {'injury_missed': 13, 'injury_types': ['Achilles']},
        {'injury_missed': 5, 'injury_types': ['Hamstring']},
        {'injury_missed': 7, 'injury_types': ['Hamstring', 'Ankle']}
    ]
    classification, multiplier = classify_injury_pattern(prone_history)
    assert classification == 'injury-prone'
    assert multiplier == 1.00

def test_effective_games_calculation():
    """Test effective games adjustment"""
    
    # Reliable player: played 10, missed 7
    # Should get significant benefit
    effective = calculate_injury_adjusted_games(
        "mock_id", 2024, games_played=10, max_games=17
    )
    assert effective > 10  # Should be adjusted upward
    assert effective <= 17
```

### Test 2: Integration Test - Full Pipeline

**Test**: Run `main.py` for 2024 season and verify:

1. **Injury data loads successfully**
2. **Phase 5 applies injury adjustments**
3. **Rankings reflect injury-adjusted scores**
4. **Known cases produce expected results**:
   - Derrick Henry (durable) should have high effective games
   - CMC (injury-prone) should have lower effective games
   - One-off injury players should be in middle

### Test 3: Regression Test - Compare Before/After

**Process**:
1. Run pipeline WITHOUT injury adjustment (current state)
2. Save 2024 rankings
3. Implement injury adjustment
4. Run pipeline WITH injury adjustment
5. Compare rankings - verify changes make sense

**Expected Changes**:
- Durable players who missed 1-2 games: Minimal rank change
- Injury-prone players: May drop in rankings (higher dampening)
- One-off injury cases: Moderate adjustment

---

## Performance Considerations

### Data Loading

**Issue**: Loading injury/roster data adds API calls and processing time

**Solutions**:
1. ✅ **Cache data locally** - Already implemented in loader functions
2. **Lazy loading** - Only load data when needed (first Phase 5 call)
3. **Batch loading** - Load all years upfront in one nflreadpy call

### Optimization Options

**Option A: Pre-compute Injury History** (Recommended)
```python
# At start of main.py, before processing any years
def preload_injury_data(start_year, end_year):
    """Load all injury data upfront"""
    logger.info("Preloading injury and roster data...")
    
    for year in range(start_year, end_year + 1):
        load_injury_data(year)  # Will cache if not present
        load_roster_data(year)
    
    logger.info("Injury data preload complete")

# Call before year loop
preload_injury_data(args.start_year - 2, args.end_year)
```

**Option B: Parallel Processing**
```python
# Load multiple years in parallel
from concurrent.futures import ThreadPoolExecutor

def parallel_load_injury_data(years):
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(load_injury_data, years)
        executor.map(load_roster_data, years)
```

---

## Error Handling

### Missing GSIS IDs

**Problem**: Player not found in roster data

**Solution**:
```python
if not gsis_id:
    logger.warning(f"No GSIS ID for {player_name}, using actual games played")
    return float(games_played)  # No adjustment
```

### Incomplete Injury Data

**Problem**: Injury or roster data missing for a season

**Solution**:
```python
if len(injury_history) == 0:
    logger.warning(f"No injury history for {gsis_id}, using actual games")
    return float(games_played)
```

### API Failures

**Problem**: nflreadpy API down or rate-limited

**Solution**:
```python
try:
    injuries = nfl.load_injuries(seasons=season)
except Exception as e:
    logger.error(f"API failure: {e}, falling back to no adjustment")
    return float(games_played)
```

---

## Validation Checklist

Before deploying to production:

- [ ] Cache directories created (`cache/injuries/`, `cache/rosters/`)
- [ ] All constants added to `constants.py`
- [ ] `injury_cache_builder.py` created and imports work
- [ ] Phase 5 modified to call injury adjustment
- [ ] Cache build functions integrated into `rebuild_caches.py`
- [ ] Unit tests pass
- [ ] Integration test with 2024 data successful
- [ ] Known cases produce expected results (CMC, Henry, etc.)
- [ ] Performance acceptable (< 2x current runtime)
- [ ] Error handling tested (missing IDs, incomplete data)
- [ ] Documentation updated in README

---

## Rollout Plan

### Phase A: Development (Current)
1. Create `injury_cache_builder.py` module
2. Add constants
3. Implement core functions
4. Write unit tests

### Phase B: Integration (Next)
1. Modify Phase 5 in `main.py`
2. Run test on 2024 season only
3. Verify results manually
4. Compare before/after rankings

### Phase C: Full Deployment
1. Preload injury data for 2016-2025
2. Run full pipeline with injury adjustment
3. Validate output files
4. Document changes in git commit

### Phase D: Monitoring
1. Check for edge cases in output
2. Verify no players have `NaN` scores
3. Confirm ranking distributions look reasonable
4. Monitor for any nflreadpy API issues

---

## Example Usage

### Manual Testing

```python
# Test injury cache builder on known player
from modules.injury_cache_builder import calculate_injury_adjusted_games

# CMC in 2024 (played 4 games, missed 13)
effective = calculate_injury_adjusted_games(
    player_gsis_id="00-0033908",  # CMC's actual ID
    current_season=2024,
    games_played=4,
    max_games=17
)

print(f"CMC effective games: {effective}")
# Expected: ~4-6 (injury-prone classification, minimal benefit)

# Derrick Henry in 2024 (played 17 games)
effective = calculate_injury_adjusted_games(
    player_gsis_id="00-0031546",  # Henry's actual ID
    current_season=2024,
    games_played=17,
    max_games=17
)

print(f"Henry effective games: {effective}")
# Expected: 17.0 (no adjustment needed, played all games)
```

### Full Pipeline Test

```bash
# Run with injury adjustment enabled
python main.py --start-year 2024 --end-year 2024

# Check output
cat output/2024/rb_rankings.md | head -20
```

---

## Future Enhancements

### Phase 3: Advanced Injury Classification

1. **Injury severity tracking**
   - Distinguish IR (8+ weeks) from week-to-week
   - Weight long-term injuries more heavily

2. **Position-specific injury risk**
   - RBs: Weight soft tissue injuries more
   - WRs: Weight hand/finger injuries less
   - QBs: Weight concussions more

3. **Age-adjusted injury risk**
   - Older players (30+) may be more injury-prone
   - Apply additional penalty for age + injury history

### Phase 4: Real-Time Updates

1. **Weekly injury report integration**
   - Update injury data mid-season
   - Adjust projections based on current injury status

2. **Injury trend detection**
   - Flag players with worsening injury patterns
   - Alert when reliable player has concerning injury

---

## Dependencies

### Required Packages

```python
# Already in requirements.txt
nflreadpy>=0.1.0
polars>=0.20.0

# No new dependencies needed
```

### Data Availability

- `nflreadpy.load_injuries()`: 2009-present ✅
- `nflreadpy.load_rosters_weekly()`: 2002-present ✅
- Our data range: 2016-2025 ✅ (fully covered)

---

## Success Criteria

### Quantitative Metrics

1. **Accuracy**: Injury-prone players (7+ expected missed games) should have lower effective games
2. **Fairness**: One-off injury cases should not be over-penalized
3. **Performance**: Phase 5 runtime increase < 30%
4. **Coverage**: > 95% of players have GSIS IDs and injury data

### Qualitative Checks

1. Rankings "feel right" for known cases
2. No weird edge cases (NaN scores, negative values)
3. Injury classifications match intuition (CMC = prone, Henry = reliable)
4. Documentation clear enough for future modifications

---

## Contact for Questions

- Implementation questions: See this document
- Data issues: Check nflreadpy documentation
- Logic questions: Refer to `research/injury_data_analysis.md`

---

## Appendix: GSIS ID Mapping

### Finding GSIS IDs

**Method 1: From existing data**
```python
# Your PBP data already has GSIS IDs
pbp = pl.read_parquet('pbp_data.parquet')
player_ids = pbp.select(['player_name', 'gsis_player_id']).unique()
```

**Method 2: From nflreadpy rosters**
```python
rosters = nfl.load_rosters_weekly(seasons=2024)
cmc = rosters.filter(pl.col('full_name') == 'Christian McCaffrey')
print(cmc.select('gsis_id'))
```

**Method 3: From nflreadpy players**
```python
players = nfl.load_players()
cmc = players.filter(pl.col('display_name') == 'Christian McCaffrey')
print(cmc.select('gsis_id'))
```

### Common GSIS IDs (for testing)

```python
TEST_PLAYERS = {
    'Christian McCaffrey': '00-0033908',
    'Derrick Henry': '00-0031546',
    'Saquon Barkley': '00-0034844',
    'Austin Ekeler': '00-0033040',
}
```

---

## Implementation Timeline

**Estimated Time**: 4-6 hours

1. **Hour 1**: Create `injury_cache_builder.py` module (core functions)
2. **Hour 2**: Add constants, implement GSIS ID mapping
3. **Hour 3**: Modify Phase 5 in `main.py`
4. **Hour 4**: Test on 2024 season, debug issues
5. **Hour 5**: Write unit tests, validate results
6. **Hour 6**: Run full 2016-2025 pipeline, document

---

## End of Implementation Plan

This document should be sufficient for any AI agent with repository access to implement the injury adjustment system seamlessly. All functions, integration points, and validation steps are documented in detail.
