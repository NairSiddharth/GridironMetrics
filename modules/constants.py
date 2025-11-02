from datetime import datetime
from pathlib import Path

# ============================================================================
# TEAM CODE MAPPINGS AND RELOCATIONS
# ============================================================================

# Pro-Football-Reference team codes (used for PFR scraping)
# NOTE: PFR uses static codes that never change, even after relocations
TEAM_CODES = {
    "crd": "Arizona Cardinals",
    "atl": "Atlanta Falcons",
    "rav": "Baltimore Ravens",
    "buf": "Buffalo Bills",
    "car": "Carolina Panthers",
    "chi": "Chicago Bears",
    "cin": "Cincinnati Bengals",
    "cle": "Cleveland Browns",
    "dal": "Dallas Cowboys",
    "den": "Denver Broncos",
    "det": "Detroit Lions",
    "gnb": "Green Bay Packers",
    "htx": "Houston Texans",
    "clt": "Indianapolis Colts",
    "jax": "Jacksonville Jaguars",
    "kan": "Kansas City Chiefs",
    "rai": "Las Vegas Raiders",
    "sdg": "Los Angeles Chargers",
    "ram": "Los Angeles Rams",
    "mia": "Miami Dolphins",
    "min": "Minnesota Vikings",
    "nwe": "New England Patriots",
    "nor": "New Orleans Saints",
    "nyg": "New York Giants",
    "nyj": "New York Jets",
    "phi": "Philadelphia Eagles",
    "pit": "Pittsburgh Steelers",
    "sfo": "San Francisco 49ers",
    "sea": "Seattle Seahawks",
    "tam": "Tampa Bay Buccaneers",
    "oti": "Tennessee Titans",
    "was": "Washington Commanders",
}

# nflverse team codes (used in cache directories and data files)
# Includes both current and historical codes for relocated teams
NFLVERSE_TEAM_CODES = {
    # Current teams (as of 2025)
    "ari": "Arizona Cardinals",
    "atl": "Atlanta Falcons",
    "bal": "Baltimore Ravens",
    "buf": "Buffalo Bills",
    "car": "Carolina Panthers",
    "chi": "Chicago Bears",
    "cin": "Cincinnati Bengals",
    "cle": "Cleveland Browns",
    "dal": "Dallas Cowboys",
    "den": "Denver Broncos",
    "det": "Detroit Lions",
    "gb": "Green Bay Packers",
    "hou": "Houston Texans",
    "ind": "Indianapolis Colts",
    "jax": "Jacksonville Jaguars",
    "kc": "Kansas City Chiefs",
    "lv": "Las Vegas Raiders",          # Current (2020+)
    "lac": "Los Angeles Chargers",      # Current (2017+)
    "la": "Los Angeles Rams",           # Current (2016+)
    "mia": "Miami Dolphins",
    "min": "Minnesota Vikings",
    "ne": "New England Patriots",
    "no": "New Orleans Saints",
    "nyg": "New York Giants",
    "nyj": "New York Jets",
    "phi": "Philadelphia Eagles",
    "pit": "Pittsburgh Steelers",
    "sf": "San Francisco 49ers",
    "sea": "Seattle Seahawks",
    "tb": "Tampa Bay Buccaneers",
    "ten": "Tennessee Titans",
    "was": "Washington Commanders",

    # Historical codes (for relocated teams, used in early cache data 2000-2002)
    "oak": "Oakland Raiders",           # Historical (used 2000-2002 in cache)
    "sd": "San Diego Chargers",         # Historical (used 2000-2002 in cache)
    "stl": "St. Louis Rams",            # Historical (used 2000-2002 in cache)
    "jac": "Jacksonville Jaguars",      # Historical alternate code
}

# Team relocations mapping: tracks when teams moved and what codes changed
TEAM_RELOCATIONS = {
    "sd": {
        "new_code": "lac",
        "relocation_year": 2017,
        "old_location": "San Diego",
        "new_location": "Los Angeles",
        "franchise": "Chargers"
    },
    "oak": {
        "new_code": "lv",
        "relocation_year": 2020,
        "old_location": "Oakland",
        "new_location": "Las Vegas",
        "franchise": "Raiders"
    },
    "stl": {
        "new_code": "la",
        "relocation_year": 2016,
        "old_location": "St. Louis",
        "new_location": "Los Angeles",
        "franchise": "Rams"
    }
}

# Pro-Football-Reference codes → nflverse codes mapping
# PFR uses static codes, nflverse uses current location codes retroactively
PFR_TO_NFLVERSE_MAP = {
    # Standard mappings (different abbreviation styles)
    "crd": "ari",  # Arizona Cardinals
    "rav": "bal",  # Baltimore Ravens
    "clt": "ind",  # Indianapolis Colts
    "gnb": "gb",   # Green Bay Packers
    "htx": "hou",  # Houston Texans
    "kan": "kc",   # Kansas City Chiefs
    "nwe": "ne",   # New England Patriots
    "nor": "no",   # New Orleans Saints
    "oti": "ten",  # Tennessee Titans
    "sfo": "sf",   # San Francisco 49ers
    "tam": "tb",   # Tampa Bay Buccaneers

    # Relocated teams (PFR uses historical codes, nflverse uses current)
    "rai": "lv",   # Raiders (PFR: rai for all years, nflverse: lv)
    "sdg": "lac",  # Chargers (PFR: sdg for all years, nflverse: lac)
    "ram": "la",   # Rams (PFR: ram for all years, nflverse: la)

    # Teams with same codes in both systems
    "atl": "atl", "buf": "buf", "car": "car", "chi": "chi", "cin": "cin",
    "cle": "cle", "dal": "dal", "den": "den", "det": "det", "jax": "jax",
    "mia": "mia", "min": "min", "nyg": "nyg", "nyj": "nyj", "phi": "phi",
    "pit": "pit", "sea": "sea", "was": "was"
}

# Reverse mapping: nflverse → PFR
NFLVERSE_TO_PFR_MAP = {v: k for k, v in PFR_TO_NFLVERSE_MAP.items()}

START_YEAR = 2000  # change if you want earlier data
END_YEAR = datetime.now().year   # change if you want later data

# Use absolute path for cache directory (resolves relative to project root)
CACHE_DIR = str((Path(__file__).parent.parent / "cache").resolve())

# Games per season by year (for normalization)
GAMES_PER_SEASON = {
    # Modern era
    **{year: 17 for year in range(2021, 2030)},  # 2021 onwards: 17 games
    **{year: 16 for year in range(1978, 2021)},  # 1978-2020: 16 games
    # Historical
    1987: 15,  # Strike year
    1982: 9,   # Strike year
    **{year: 16 for year in range(1961, 1978)},  # 1961-1977: 16 games (correcting from table)
    **{year: 14 for year in range(1961, 1978)},  # 1961-1977: 14 games
    **{year: 12 for year in range(1947, 1961)},  # 1947-1960: 12 games
    1946: 11,
    1945: 10,
    1944: 9,
    1943: 8,
    **{year: 11 for year in range(1937, 1943)},  # 1937-1942: 11 games
    **{year: 12 for year in range(1935, 1937)},  # 1935-1936: 12 games
    **{year: 11 for year in range(1925, 1935)},  # 1925-1934: 11 games
    **{year: 8 for year in range(1920, 1925)},   # 1920-1924: 8 games
}

# Rate limiting and jail policy
# Max requests per minute for strict domains (FBref/Stathead/Sports-Reference family)
RATE_LIMIT_STRICT = 10
# Max requests per minute for other domains
RATE_LIMIT_OTHER = 20
# Default jail duration in seconds when a session/domain is put in jail for violations
JAIL_DURATION_SECONDS = 24 * 60 * 60  # 24 hours

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

# ============================================================================
# PENALTY ADJUSTMENT CONSTANTS (Phase 4)
# ============================================================================

# Penalty Type Classifications - Base Multipliers
PENALTY_LOSS_OF_DOWN_BASE = 2.0      # Intentional grounding, illegal forward pass
PENALTY_DEAD_BALL_BASE = 1.5         # Taunting, unsportsmanlike conduct
PENALTY_REPEAT_OFFENDER_BASE = 2.0   # Dead ball penalties when player has 2+
PENALTY_STANDARD_BASE = 1.0          # OPI, face mask, unnecessary roughness

# Down Situation Multipliers
PENALTY_DOWN_1ST = 1.0               # 1st down - baseline
PENALTY_DOWN_2ND = 1.2               # 2nd down - more critical
PENALTY_DOWN_3RD_4TH = 1.5           # 3rd/4th down - drive killer

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

# ============================================================================
# WEATHER ADJUSTMENT CONSTANTS (Phase 4.5)
# ============================================================================

# Temperature Thresholds (Fahrenheit)
WEATHER_TEMP_COLD = 32.0      # Below 32°F = cold
WEATHER_TEMP_COOL = 55.0      # 32-55°F = cool
WEATHER_TEMP_MODERATE = 75.0  # 55-75°F = moderate
# Above 75°F = hot

# Wind Thresholds (mph)
WEATHER_WIND_CALM = 10.0      # Below 10 mph = calm
WEATHER_WIND_MODERATE = 20.0  # 10-20 mph = moderate
# Above 20 mph = high

# Adjustment Factor Caps
WEATHER_FACTOR_MIN = 0.95     # Min adjustment per individual factor
WEATHER_FACTOR_MAX = 1.05     # Max adjustment per individual factor
WEATHER_TOTAL_MIN = 0.90      # Min total weather adjustment
WEATHER_TOTAL_MAX = 1.10      # Max total weather adjustment

# Sample Size Requirements
WEATHER_MIN_PLAYS = 30        # Minimum plays in a condition to calculate adjustment

# Precipitation Keywords (for weather description parsing)
WEATHER_PRECIP_KEYWORDS = [
    'rain', 'snow', 'sleet', 'hail', 'drizzle',
    'showers', 'flurries', 'precipitation'
]

# ============================================================================
# TEAM CODE UTILITY FUNCTIONS
# ============================================================================

def get_team_code_for_year(team_code: str, year: int) -> str:
    """
    Get the appropriate team code for a given year, accounting for relocations.

    This function handles the complexity of team relocations and different coding
    systems (PFR vs nflverse). It returns the correct nflverse code to use for
    loading data from cache directories.

    Args:
        team_code: Team code (can be old/new, PFR/nflverse format)
        year: Season year

    Returns:
        Correct nflverse team code for that year

    Examples:
        >>> get_team_code_for_year('sd', 2015)   # San Diego Chargers before move
        'sd'
        >>> get_team_code_for_year('sd', 2018)   # After relocation to LA
        'lac'
        >>> get_team_code_for_year('lac', 2015)  # Asking for LAC in 2015
        'sd'
        >>> get_team_code_for_year('rai', 2024)  # PFR code for Raiders
        'lv'
    """
    # Normalize to lowercase
    team_code = team_code.lower()

    # First, convert PFR codes to nflverse codes
    if team_code in PFR_TO_NFLVERSE_MAP:
        team_code = PFR_TO_NFLVERSE_MAP[team_code]

    # For years 2000-2002, nflverse cache uses old codes
    # (due to data inconsistency in early nflverse processing)
    if year <= 2002:
        # If this is a new code for a relocated team, use the old code instead
        for old_code, info in TEAM_RELOCATIONS.items():
            if team_code == info["new_code"]:
                return old_code
        return team_code

    # For years 2003+, check if this team has relocated
    # Use the appropriate code based on relocation year
    for old_code, info in TEAM_RELOCATIONS.items():
        new_code = info["new_code"]
        relocation_year = info["relocation_year"]

        # If we're asking about this franchise (old or new code)
        if team_code in [old_code, new_code]:
            # nflverse retroactively uses new codes for ALL years 2003+
            # regardless of actual relocation year
            return new_code

    # No relocation applies, return as-is
    return team_code


def get_all_team_codes_for_year(year: int) -> list:
    """
    Get all valid team codes for a specific year.

    This accounts for:
    - Teams that didn't exist yet (e.g., Houston Texans before 2002)
    - Team relocations and which codes to use

    Args:
        year: Season year

    Returns:
        List of team codes that should have data for that year
    """
    all_codes = []

    # Get all nflverse codes
    for code in NFLVERSE_TEAM_CODES.keys():
        # Skip historical codes if we're past the relocation
        if code in ["oak", "sd", "stl"] and year > 2002:
            continue

        # Skip current relocated team codes for early years
        if year <= 2002:
            if code in ["lv", "lac", "la"]:
                continue

        # Skip Houston Texans before 2002
        if code == "hou" and year < 2002:
            continue

        # Skip jac (use jax instead)
        if code == "jac":
            continue

        all_codes.append(code)

    return sorted(all_codes)


def normalize_team_code(team_code: str, source: str = "auto") -> str:
    """
    Normalize a team code to nflverse format.

    Args:
        team_code: Team code to normalize
        source: Source system - "pfr", "nflverse", or "auto" (detect automatically)

    Returns:
        Normalized nflverse team code
    """
    team_code = team_code.lower()

    if source == "pfr" or (source == "auto" and team_code in PFR_TO_NFLVERSE_MAP):
        return PFR_TO_NFLVERSE_MAP.get(team_code, team_code)

    return team_code


def normalize_team_codes_in_dataframe(df, year: int, team_column: str = "team",
                                      opponent_column: str = "opponent_team"):
    """
    Normalize team codes in a Polars DataFrame to handle relocations.

    For team and opponent columns, ensures consistent codes accounting for:
    - Historical relocations (SD→LAC, OAK→LV, STL→LA)
    - nflverse retroactive standardization
    - Pro-Football-Reference static codes

    Args:
        df: Polars DataFrame with team code columns
        year: Season year (determines which codes to use)
        team_column: Name of the team column (default: "team")
        opponent_column: Name of the opponent team column (default: "opponent_team")

    Returns:
        DataFrame with normalized team codes

    Note:
        This function uses vectorized operations for thread-safety and performance.
        Previous implementation with map_elements caused deadlocks in parallel processing.
    """
    try:
        import polars as pl
    except ImportError:
        # If polars not available, return unchanged
        return df

    # Build replacement mapping for this specific year (vectorized)
    # This avoids per-row lambda calls that cause GIL contention in threads
    replacements = {}

    # Handle relocations based on year
    if year > 2016:
        replacements['sd'] = 'lac'
        replacements['SD'] = 'LAC'
    else:
        replacements['lac'] = 'sd'
        replacements['LAC'] = 'SD'

    if year > 2019:
        replacements['oak'] = 'lv'
        replacements['OAK'] = 'LV'
    else:
        replacements['lv'] = 'oak'
        replacements['LV'] = 'OAK'

    if year > 2015:
        replacements['stl'] = 'la'
        replacements['STL'] = 'LA'
    else:
        replacements['la'] = 'stl'
        replacements['LA'] = 'STL'

    # Normalize team column if it exists (vectorized replace)
    if team_column in df.columns:
        df = df.with_columns([
            pl.col(team_column).replace(replacements, default=pl.col(team_column)).alias(team_column)
        ])

    # Normalize opponent column if it exists (vectorized replace)
    if opponent_column in df.columns:
        df = df.with_columns([
            pl.col(opponent_column).replace(replacements, default=pl.col(opponent_column)).alias(opponent_column)
        ])

    return df

# ============================================================================
# METRIC CONFIGURATIONS
# ============================================================================

# Metric groupings for combined stats
COMBINED_METRICS = {
    'total_yards': {
        'metrics': ['receiving_yards', 'rushing_yards'],
        'display': 'Total Yards (Rush + Rec)'
    },
    'total_touchdowns': {
        'metrics': ['receiving_tds', 'rushing_tds'],
        'display': 'Total TDs (Rush + Rec)'
    },
    'overall_contribution': {
        'metrics': ['receiving_yards', 'rushing_yards', 'receiving_tds', 'rushing_tds', 'receptions', 'targets', 'carries'],
        'display': 'Overall Offensive Contribution',
        'weights': {  # Weights based on EPA and WPA analysis
            'receiving_yards': 1.0,      # Base unit for comparison
            'rushing_yards': 1.0,        # Equal to receiving yards in value
            'receiving_tds': 50.0,       # Based on EPA conversion (~7 points / 0.14 points per yard)
            'rushing_tds': 50.0,         # Equal to receiving TD in value
            'receptions': 8.0,           # Success rate and chain-moving value beyond yards
            'targets': 3.0,              # Opportunity cost and defensive attention value
            'carries': 3.0               # Similar opportunity value to targets
        }
    },
    'qb_contribution': {
        'metrics': ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds', 'completions', 'attempts'],
        'display': 'QB Overall Contribution',
        'weights': {
            'passing_yards': 1.0,        # Base unit
            'passing_tds': 50.0,         # ~7 points / 0.14 per yard
            'rushing_yards': 1.2,        # Slightly higher value for QB rushing
            'rushing_tds': 50.0,         # Equal TD value
            'completions': 5.0,          # Chain-moving and success rate value
            'attempts': -1.0             # Penalty for inefficiency (balanced by completions)
        }
    }
}

SKILL_POSITIONS = ['WR', 'RB', 'TE']

# Skill position metrics for table generation
SKILL_POSITION_METRICS = {
    'rushing_yards': 'Rush Yards',
    'receiving_yards': 'Rec Yards',
    'rushing_tds': 'Rush TD',
    'receiving_tds': 'Rec TD',
    'receptions': 'Receptions',
    'targets': 'Targets',
    'carries': 'Rush Att'
}

# QB-specific metrics
QB_METRICS = {
    'passing_yards': 'Pass Yards',
    'passing_tds': 'Pass TDs',
    'completions': 'Completions',
    'attempts': 'Attempts',
    'passing_interceptions': 'Interceptions',
    'rushing_yards': 'Rush Yards',
    'rushing_tds': 'Rush TDs'
}