from datetime import datetime
from pathlib import Path

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