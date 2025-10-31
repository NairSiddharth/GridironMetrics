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
    INJURY_PENALTY_RELIABLE,
    INJURY_PENALTY_MODERATE,
    INJURY_PENALTY_ELEVATED,
    INJURY_PENALTY_PRONE,
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
    
    # Count injury-related misses
    # - RES (Reserve/IR) is ALWAYS injury-related
    # - INA (Inactive) requires injury report to confirm it's injury vs healthy scratch
    injury_missed_df = combined.filter(
        (pl.col('status') == 'RES') |  # IR always counts as injury
        ((pl.col('status') == 'INA') & (pl.col('report_status') == 'Out'))  # INA + Out report
    )
    injury_missed = len(injury_missed_df)
    
    # Count non-injury inactives (INA without injury report)
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
        - classification: 'reliable', 'moderate', 'elevated', 'injury-prone'
        - multiplier: Adjustment factor for effective games calculation
    """
    if not injury_history or len(injury_history) == 0:
        return ('reliable', INJURY_PENALTY_RELIABLE)
    
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
        multiplier = INJURY_PENALTY_RELIABLE
    elif expected_missed < INJURY_MODERATE_THRESHOLD:
        classification = 'moderate'
        multiplier = INJURY_PENALTY_MODERATE
    elif expected_missed < INJURY_PRONE_THRESHOLD:
        classification = 'elevated'
        multiplier = INJURY_PENALTY_ELEVATED
    else:
        classification = 'injury-prone'
        multiplier = INJURY_PENALTY_PRONE
    
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
        player_name: Player name (may be "F.LastName" or "Full Name")
        team: Team abbreviation (e.g., "SF")
        position: Position code (e.g., "RB")
        season: Season year
        
    Returns:
        GSIS ID string, or empty string if not found
    """
    rosters = load_roster_data(season)
    
    if rosters.is_empty():
        logger.error(f"No roster data available for {season}")
        return ""
    
    # Try exact match first (for full names)
    match = rosters.filter(
        (pl.col('full_name') == player_name) &
        (pl.col('team') == team) &
        (pl.col('position') == position)
    )
    
    if len(match) > 0:
        return match.select('gsis_id').item(0, 0)
    
    # Handle "F.LastName" format - extract last name
    if '.' in player_name:
        last_name = player_name.split('.')[-1]
    else:
        # Full name - use last word as last name
        last_name = player_name.split()[-1]
    
    # Try matching by last name, team, and position
    match = rosters.filter(
        (pl.col('last_name') == last_name) &
        (pl.col('team') == team) &
        (pl.col('position') == position)
    )
    
    if len(match) > 0:
        # Get unique GSIS IDs (roster data has multiple weeks)
        unique_gsis = match.select('gsis_id').unique()
        
        if len(unique_gsis) == 1:
            # Single unique player found
            return unique_gsis.item(0, 0)
        else:
            # Multiple different players with same last name - log warning and use first
            logger.warning(
                f"Multiple different players named {last_name} on {team} at {position}, using first match"
            )
            return unique_gsis.item(0, 0)
    
    # Last resort: fuzzy match on last name only (any team/position)
    match = rosters.filter(pl.col('last_name') == last_name)
    
    if len(match) == 1:
        logger.warning(f"Fuzzy matched {player_name} to {match['full_name'].item(0)}")
        return match.select('gsis_id').item(0, 0)
    
    logger.debug(f"Could not find GSIS ID for {player_name} ({team} {position})")
    return ""


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
