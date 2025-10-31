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