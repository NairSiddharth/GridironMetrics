"""
Player Props Type Configuration

Defines prop type → adjustment mappings and configuration.
Different props require different adjustments based on what drives their outcomes.

Rationale:
- TD Props: Prioritize success rate (red zone/chain-moving efficiency)
- Yards Props: Prioritize opponent defense (matchup-driven)
- Receptions: Prioritize catch rate (target conversion efficiency)
"""

from typing import List, Dict, Optional


# ============================================================================
# PROP TYPE DEFINITIONS
# ============================================================================

PROP_TYPE_ADJUSTMENTS = {
    'passing_yards': {
        'adjustments': [
            'opponent_defense',  # Rolling opponent pass defense through week N
            'weather',           # Temperature, wind, precipitation
            'success_rate',      # Passing efficiency (3-week rolling)
        ],
        'api_market': 'player_pass_yds',
        'stat_column': 'passing_yards',
        'position': ['QB'],
        'min_sample_size': 30,  # Minimum 30 attempts
        'display_name': 'Passing Yards',
    },
    'passing_tds': {
        'adjustments': [
            'success_rate',      # Red zone / chain-moving efficiency
            'opponent_defense',  # Opponent pass TD defense
            'weather',           # Weather impact on passing
        ],
        'api_market': 'player_pass_tds',
        'stat_column': 'passing_tds',
        'position': ['QB'],
        'min_sample_size': 30,
        'display_name': 'Passing Touchdowns',
    },
    'rushing_yards': {
        'adjustments': [
            'opponent_defense',  # Opponent rush defense
            'blocking_quality',  # RB YPC vs teammate RB avg
            'weather',           # Precipitation, temperature
        ],
        'api_market': 'player_rush_yds',
        'stat_column': 'rushing_yards',
        'position': ['RB'],  # RB-only (QB rushing is too noisy/unpredictable)
        'min_sample_size': 20,  # Minimum 20 carries
        'display_name': 'Rushing Yards',
    },
    'rushing_tds': {
        'adjustments': [
            'success_rate',      # Goal-line / short yardage efficiency
            'opponent_defense',  # Opponent rush TD defense
            'route_location',    # Field position tendency
        ],
        'api_market': 'player_rush_tds',
        'stat_column': 'rushing_tds',
        'position': ['RB', 'QB'],
        'min_sample_size': 20,
        'display_name': 'Rushing Touchdowns',
    },
    'receiving_yards_wr': {
        'adjustments': [
            'opponent_defense',  # Opponent pass defense
            'catch_rate',        # Target efficiency
            'separation',        # NextGen receiver separation
            'weather',           # Weather impact
        ],
        'api_market': 'player_reception_yds',
        'stat_column': 'receiving_yards',
        'position': ['WR'],
        'min_sample_size': 40,  # Minimum 40 season targets
        'min_weekly_volume': 3,  # Minimum 3 targets per week
        'display_name': 'Receiving Yards (WR)',
    },
    'receiving_yards_te': {
        'adjustments': [
            'opponent_defense',  # Opponent pass defense
            'catch_rate',        # Target efficiency
            'separation',        # NextGen receiver separation
            'weather',           # Weather impact
        ],
        'api_market': 'player_reception_yds',
        'stat_column': 'receiving_yards',
        'position': ['TE'],
        'min_sample_size': 40,  # Minimum 40 season targets
        'min_weekly_volume': 3,  # Minimum 3 targets per week
        'display_name': 'Receiving Yards (TE)',
    },
    'receiving_tds': {
        'adjustments': [
            'success_rate',      # Red zone efficiency
            'route_location',    # Route positioning
            'opponent_defense',  # Opponent pass TD defense
        ],
        'api_market': 'player_reception_tds',
        'stat_column': 'receiving_tds',
        'position': ['WR', 'TE'],
        'min_sample_size': 15,
        'display_name': 'Receiving Touchdowns',
    },
    'receptions': {
        'adjustments': [
            'catch_rate',        # Target conversion rate
            'opponent_defense',  # Opponent pass defense
        ],
        'api_market': 'player_receptions',
        'stat_column': 'receptions',
        'position': ['WR', 'TE'],
        'min_sample_size': 15,
        'display_name': 'Receptions',
    },
}


# ============================================================================
# API MARKET MAPPINGS
# ============================================================================

# Map The Odds API market keys to internal prop types
API_MARKET_TO_PROP_TYPE = {
    'player_pass_yds': 'passing_yards',
    'player_pass_tds': 'passing_tds',
    'player_rush_yds': 'rushing_yards',
    'player_rush_tds': 'rushing_tds',
    'player_reception_yds': 'receiving_yards',
    'player_reception_tds': 'receiving_tds',
    'player_receptions': 'receptions',
}

# Reverse mapping
PROP_TYPE_TO_API_MARKET = {v: k for k, v in API_MARKET_TO_PROP_TYPE.items()}


# ============================================================================
# POSITION → PROP TYPE MAPPINGS
# ============================================================================

POSITION_PROP_TYPES = {
    'QB': [
        'passing_yards',
        'passing_tds',
        'rushing_yards',
        'rushing_tds',
    ],
    'RB': [
        'rushing_yards',
        'rushing_tds',
        'receptions',
        'receiving_yards_wr',  # RBs train with WR model (similar receiving usage)
        'receiving_tds',
    ],
    'WR': [
        'receptions',
        'receiving_yards_wr',
        'receiving_tds',
    ],
    'TE': [
        'receptions',
        'receiving_yards_te',
        'receiving_tds',
    ],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_adjustments_for_prop(prop_type: str) -> List[str]:
    """
    Get ordered list of adjustments to apply for a prop type.

    Args:
        prop_type: Prop type (e.g., 'passing_yards')

    Returns:
        List of adjustment names in application order
    """
    if prop_type not in PROP_TYPE_ADJUSTMENTS:
        return []

    return PROP_TYPE_ADJUSTMENTS[prop_type]['adjustments']


def get_api_market_for_prop(prop_type: str) -> Optional[str]:
    """
    Get The Odds API market key for a prop type.

    Args:
        prop_type: Prop type (e.g., 'passing_yards')

    Returns:
        API market key (e.g., 'player_pass_yds') or None
    """
    return PROP_TYPE_TO_API_MARKET.get(prop_type)


def get_stat_column_for_prop(prop_type: str) -> Optional[str]:
    """
    Get cache stat column name for a prop type.

    Args:
        prop_type: Prop type

    Returns:
        Column name in positional_player_stats cache
    """
    if prop_type not in PROP_TYPE_ADJUSTMENTS:
        return None

    return PROP_TYPE_ADJUSTMENTS[prop_type]['stat_column']


def get_prop_config(prop_type: str) -> Optional[Dict]:
    """
    Get complete configuration for a prop type.

    Args:
        prop_type: Prop type

    Returns:
        Configuration dict or None
    """
    return PROP_TYPE_ADJUSTMENTS.get(prop_type)


def get_prop_types_for_position(position: str) -> List[str]:
    """
    Get all prop types available for a position.

    Args:
        position: Position code (QB, RB, WR, TE)

    Returns:
        List of prop types
    """
    return POSITION_PROP_TYPES.get(position.upper(), [])


def get_display_name(prop_type: str) -> str:
    """
    Get human-readable display name for a prop type.

    Args:
        prop_type: Prop type

    Returns:
        Display name
    """
    config = get_prop_config(prop_type)
    if config:
        return config['display_name']
    return prop_type.replace('_', ' ').title()


def get_min_sample_size(prop_type: str) -> int:
    """
    Get minimum sample size requirement for a prop type.

    Args:
        prop_type: Prop type

    Returns:
        Minimum attempts/targets/carries required
    """
    config = get_prop_config(prop_type)
    if config:
        return config['min_sample_size']
    return 10  # Default


def is_position_eligible_for_prop(position: str, prop_type: str) -> bool:
    """
    Check if a position is eligible for a prop type.

    Args:
        position: Position code
        prop_type: Prop type

    Returns:
        True if eligible
    """
    config = get_prop_config(prop_type)
    if not config:
        return False

    return position.upper() in config['position']


# ============================================================================
# ADJUSTMENT RATIONALE DOCUMENTATION
# ============================================================================

ADJUSTMENT_RATIONALE = {
    'passing_yards': """
    **Why These Adjustments:**
    - Opponent Defense (1st): Passing yards heavily matchup-dependent
    - Weather (2nd): Wind/precip directly impacts passing efficiency
    - Success Rate (3rd): Efficient passers generate more opportunities
    """,

    'passing_tds': """
    **Why These Adjustments:**
    - Success Rate (1st): TDs require chain-moving to reach red zone
    - Opponent Defense (2nd): Red zone defense quality matters
    - Weather (3rd): Adverse weather impacts TD opportunities
    """,

    'rushing_yards': """
    **Why These Adjustments:**
    - Opponent Defense (1st): Run defense quality is primary driver
    - Blocking Quality (2nd): OL performance enables rushing yards
    - Weather (3rd): Precipitation favors run game
    """,

    'rushing_tds': """
    **Why These Adjustments:**
    - Success Rate (1st): Goal-line efficiency is key for TDs
    - Opponent Defense (2nd): Goal-line defense matters
    - Route Location (3rd): Field position correlates with TD opportunity
    """,

    'receiving_yards': """
    **Why These Adjustments:**
    - Opponent Defense (1st): Pass defense quality primary factor
    - Catch Rate (2nd): Target efficiency drives volume
    - Separation (3rd): Creating space enables yards after catch
    - Weather (4th): Wind/precip affects passing game
    """,

    'receiving_tds': """
    **Why These Adjustments:**
    - Success Rate (1st): Red zone efficiency creates TD opportunities
    - Route Location (2nd): Certain routes/locations more TD-prone
    - Opponent Defense (3rd): Red zone pass defense quality
    """,

    'receptions': """
    **Why These Adjustments:**
    - Catch Rate (1st): Directly drives receptions (targets × catch%)
    - Opponent Defense (2nd): Opponent coverage affects target volume
    """,
}


if __name__ == "__main__":
    # Test configuration
    print("=== Prop Types Configuration ===\n")

    print("Available Prop Types:")
    for prop_type in PROP_TYPE_ADJUSTMENTS.keys():
        print(f"  - {prop_type}")

    print("\n" + "="*60)
    print("Example: Passing Yards Configuration")
    print("="*60)

    config = get_prop_config('passing_yards')
    print(f"Display Name: {config['display_name']}")
    print(f"API Market: {config['api_market']}")
    print(f"Stat Column: {config['stat_column']}")
    print(f"Positions: {config['position']}")
    print(f"Min Sample: {config['min_sample_size']}")
    print(f"Adjustments: {config['adjustments']}")

    print("\n" + "="*60)
    print("QB Prop Types")
    print("="*60)
    qb_props = get_prop_types_for_position('QB')
    for prop in qb_props:
        print(f"  - {get_display_name(prop)}")
