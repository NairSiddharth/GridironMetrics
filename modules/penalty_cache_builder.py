"""
Penalty Cache Builder Module

Loads and caches play-by-play penalty data for skill player penalty adjustments.
Tracks offensive penalties committed by QB/RB/WR/TE with full situational context.
"""

import polars as pl
import nflreadpy as nfl
from pathlib import Path
from modules.logger import get_logger
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

# Initialize logger
logger = get_logger(__name__)

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
            # Handle null values
            if row['down'] is None or row['yardline_100'] is None or \
               row['qtr'] is None or row['quarter_seconds_remaining'] is None or \
               row['epa'] is None:
                logger.debug(f"Skipping penalty with missing data: {row['penalty_type']}")
                continue
            
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
        
        player_name = player_penalties[0]['penalty_player_name'] if len(player_penalties) > 0 else "Unknown"
        logger.info(
            f"Penalty adjustment for {player_name}: "
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
