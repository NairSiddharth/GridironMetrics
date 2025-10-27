from datetime import datetime

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

CACHE_DIR = "cache"

# Rate limiting and jail policy
# Max requests per minute for strict domains (FBref/Stathead/Sports-Reference family)
RATE_LIMIT_STRICT = 10
# Max requests per minute for other domains
RATE_LIMIT_OTHER = 20
# Default jail duration in seconds when a session/domain is put in jail for violations
JAIL_DURATION_SECONDS = 24 * 60 * 60  # 24 hours