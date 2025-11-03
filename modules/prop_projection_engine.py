"""
Player Props Projection Engine

Applies Phase 1-5 adjustments to baseline projections to generate final prop projections.

Architecture:
1. Load baseline projection (rolling average from prop_data_aggregator)
2. Apply season-wide adjustments (catch rate, blocking, separation, route location)
3. Apply per-week adjustments (success rate, opponent defense, weather, penalties, turnovers)
4. Apply sample size dampening (0.4 root curve with injury adjustment)

Integration:
- Uses existing adjustment functions from adjustment_pipeline.py, context_adjustments.py, etc.
- No code duplication - references existing calculation infrastructure
"""

import polars as pl
from pathlib import Path
from typing import Dict, Tuple, Optional
from modules.logger import get_logger
from modules.constants import CACHE_DIR
from modules.prop_data_aggregator import get_player_baseline_projections, get_player_stat_summary
from modules.prop_types import get_adjustments_for_prop, get_stat_column_for_prop, get_prop_config
from modules.context_adjustments import ContextAdjustments
from modules.injury_cache_builder import calculate_injury_adjusted_games

logger = get_logger(__name__)

# Initialize context adjustments
context_adj = ContextAdjustments()


def generate_projection(
    player_id: str,
    season: int,
    week: int,
    position: str,
    prop_type: str,
    opponent_team: Optional[str] = None,
    game_weather: Optional[Dict] = None
) -> Dict:
    """
    Generate complete prop projection for a player.

    Full workflow:
    1. Load baseline projection (weighted rolling average)
    2. Apply adjustments based on prop type configuration
    3. Apply sample size dampening
    4. Return projection with breakdown

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Week to project (uses data through week-1)
        position: Player position
        prop_type: Prop type (e.g., 'passing_yards')
        opponent_team: Opponent team abbreviation (optional, for opponent defense)
        game_weather: Weather dict (temp, wind, weather, roof) (optional)

    Returns:
        {
            'player_id': str,
            'season': int,
            'week': int,
            'position': str,
            'prop_type': str,
            'baseline': float,
            'adjustments': {
                'adjustment_name': float (multiplier),
                ...
            },
            'adjusted_projection': float (before dampening),
            'effective_games': float,
            'final_projection': float (after dampening),
            'stat_summary': {
                'last_3_avg': float,
                'last_5_avg': float,
                'season_avg': float,
                'variance': float,
                'games_played': int
            }
        }
    """
    logger.info(f"Generating projection for {player_id} {prop_type} week {week}")

    # Get baseline projection (weighted rolling average through week-1)
    through_week = week - 1  # Project next week using data through current week
    baselines = get_player_baseline_projections(player_id, season, through_week, position)

    stat_column = get_stat_column_for_prop(prop_type)
    if not stat_column:
        logger.error(f"Unknown prop type: {prop_type}")
        return {}

    if stat_column not in baselines:
        logger.warning(f"No baseline found for {stat_column}")
        return {}

    baseline = baselines[stat_column]
    games_played = baselines.get('games_played', 0)

    if baseline == 0:
        logger.warning(f"Baseline is zero for {player_id} {prop_type}")
        # Return zero projection but still populate structure
        return {
            'player_id': player_id,
            'season': season,
            'week': week,
            'position': position,
            'prop_type': prop_type,
            'baseline': 0.0,
            'adjustments': {},
            'adjusted_projection': 0.0,
            'effective_games': games_played,
            'final_projection': 0.0,
            'stat_summary': get_player_stat_summary(player_id, season, through_week, position, stat_column)
        }

    # Get adjustments to apply for this prop type
    adjustments_list = get_adjustments_for_prop(prop_type)

    # Apply adjustments
    adjusted_value = baseline
    adjustment_breakdown = {}

    for adjustment_name in adjustments_list:
        multiplier = apply_adjustment(
            player_id=player_id,
            season=season,
            week=week,
            through_week=through_week,
            position=position,
            adjustment_name=adjustment_name,
            prop_type=prop_type,
            opponent_team=opponent_team,
            game_weather=game_weather
        )

        adjusted_value *= multiplier
        adjustment_breakdown[adjustment_name] = multiplier

        logger.debug(f"  {adjustment_name}: {multiplier:.3f}x → {adjusted_value:.2f}")

    # Calculate effective games for confidence grading (NOT for projection dampening)
    # Sample size dampening was designed for composite rankings, not raw stat projections
    # Applying it here systematically underestimated projections by 15-30%
    effective_games = calculate_injury_adjusted_games(
        player_gsis_id=player_id,
        current_season=season,
        games_played=games_played,
        max_games=17
    )

    # Use adjusted value directly as final projection (no dampening)
    final_projection = adjusted_value

    logger.info(
        f"  Projection complete: baseline={baseline:.2f}, "
        f"adjusted={adjusted_value:.2f}, final={final_projection:.2f} "
        f"(effective_games={effective_games:.1f})"
    )

    # Get statistical summary
    stat_summary = get_player_stat_summary(player_id, season, through_week, position, stat_column)

    return {
        'player_id': player_id,
        'season': season,
        'week': week,
        'position': position,
        'prop_type': prop_type,
        'baseline': baseline,
        'adjustments': adjustment_breakdown,
        'adjusted_projection': adjusted_value,
        'effective_games': effective_games,
        'final_projection': final_projection,  # No longer dampened
        'stat_summary': stat_summary
    }


def apply_adjustment(
    player_id: str,
    season: int,
    week: int,
    through_week: int,
    position: str,
    adjustment_name: str,
    prop_type: str,
    opponent_team: Optional[str],
    game_weather: Optional[Dict]
) -> float:
    """
    Apply a specific adjustment and return multiplier.

    This is a dispatcher function that calls the appropriate existing
    adjustment function based on the adjustment_name.

    Args:
        player_id: Player GSIS ID
        season: Season year
        week: Week to project
        through_week: Calculate adjustments through this week
        position: Player position
        adjustment_name: Name of adjustment (e.g., 'opponent_defense', 'weather')
        prop_type: Prop type
        opponent_team: Opponent team
        game_weather: Weather dict

    Returns:
        Multiplier (e.g., 1.05 = 5% boost, 0.95 = 5% penalty)
    """
    # For now, return neutral multiplier (1.0) as placeholder
    # We'll implement actual adjustment calls in next iteration
    # This allows us to test the structure first

    if adjustment_name == 'opponent_defense':
        return apply_opponent_defense_adjustment(
            player_id, season, week, through_week, position, prop_type, opponent_team
        )
    elif adjustment_name == 'weather':
        return apply_weather_adjustment(
            player_id, season, position, prop_type, game_weather
        )
    elif adjustment_name == 'success_rate':
        return apply_success_rate_adjustment(
            player_id, season, week, through_week, position, prop_type
        )
    elif adjustment_name == 'catch_rate':
        return apply_catch_rate_adjustment(
            player_id, season, through_week, position
        )
    elif adjustment_name == 'blocking_quality':
        return apply_blocking_quality_adjustment(
            player_id, season, through_week
        )
    elif adjustment_name == 'separation':
        return apply_separation_adjustment(
            player_id, season, through_week, position
        )
    elif adjustment_name == 'route_location':
        return apply_route_location_adjustment(
            player_id, season, through_week, position
        )
    elif adjustment_name == 'penalties':
        return apply_penalty_adjustment(
            player_id, season, week, through_week
        )
    elif adjustment_name == 'turnovers':
        return apply_turnover_adjustment(
            player_id, season, week, through_week, position
        )
    else:
        logger.warning(f"Unknown adjustment: {adjustment_name}")
        return 1.0  # Neutral multiplier


# ============================================================================
# ADJUSTMENT IMPLEMENTATION STUBS
# These will integrate with existing adjustment functions
# ============================================================================

def apply_opponent_defense_adjustment(
    player_id: str,
    season: int,
    week: int,
    through_week: int,
    position: str,
    prop_type: str,
    opponent_team: Optional[str]
) -> float:
    """
    Apply opponent defense adjustment based on rolling defensive rankings.

    Calculates opponent's defensive performance against this prop type
    through week N-1, then applies multiplier based on rankings.

    Args:
        opponent_team: Opponent team abbreviation (e.g., 'KC', 'BUF')

    Returns:
        Multiplier: 1.15x (worst defense) to 0.85x (best defense)
    """
    if not opponent_team:
        return 1.0  # No opponent info

    try:
        # Load play-by-play data
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            logger.warning(f"PBP file not found: {pbp_file}")
            return 1.0

        pbp = pl.read_parquet(pbp_file)

        # Filter to games through week N-1 (no future data)
        pbp_filtered = pbp.filter(pl.col('week') < week)

        # Calculate defensive stats by team based on prop type
        if 'passing' in prop_type:
            # Pass defense: yards/attempt allowed
            defense_stats = pbp_filtered.filter(
                (pl.col('defteam') == opponent_team) &
                (pl.col('pass_attempt') == 1)
            ).group_by('defteam').agg([
                pl.col('passing_yards').sum().alias('yards_allowed'),
                pl.col('pass_attempt').sum().alias('attempts_faced')
            ])

            if len(defense_stats) == 0:
                return 1.0

            yards_per_attempt = defense_stats['yards_allowed'][0] / max(defense_stats['attempts_faced'][0], 1)

            # League average ~7.0 yards/attempt
            # Tier 2: Moderate ranges ±17.5% (more conservative after baseline improvements)
            # Better defense (< 6.5) = harder matchup = lower multiplier
            # Worse defense (> 7.5) = easier matchup = higher multiplier
            if yards_per_attempt < 5.5:
                return 0.825  # Elite defense (top 3 defenses)
            elif yards_per_attempt < 6.0:
                return 0.875  # Great defense (top 10)
            elif yards_per_attempt < 6.5:
                return 0.925  # Good defense (top 15)
            elif yards_per_attempt < 7.5:
                return 1.0    # Average defense
            elif yards_per_attempt < 8.0:
                return 1.075  # Poor defense (bottom 15)
            elif yards_per_attempt < 8.5:
                return 1.125  # Bad defense (bottom 10)
            else:
                return 1.175  # Terrible defense (bottom 3)

        elif 'rushing' in prop_type:
            # Run defense: yards/carry allowed
            defense_stats = pbp_filtered.filter(
                (pl.col('defteam') == opponent_team) &
                (pl.col('rush_attempt') == 1)
            ).group_by('defteam').agg([
                pl.col('rushing_yards').sum().alias('yards_allowed'),
                pl.col('rush_attempt').sum().alias('carries_faced')
            ])

            if len(defense_stats) == 0:
                return 1.0

            yards_per_carry = defense_stats['yards_allowed'][0] / max(defense_stats['carries_faced'][0], 1)

            # League average ~4.3 yards/carry
            # Tier 2: Moderate ranges ±17.5%
            if yards_per_carry < 3.5:
                return 0.825  # Elite run defense
            elif yards_per_carry < 3.8:
                return 0.875  # Great run defense
            elif yards_per_carry < 4.1:
                return 0.925  # Good run defense
            elif yards_per_carry < 4.6:
                return 1.0    # Average
            elif yards_per_carry < 5.0:
                return 1.075  # Poor run defense
            elif yards_per_carry < 5.5:
                return 1.125  # Bad run defense
            else:
                return 1.175  # Terrible run defense

        elif 'receiving' in prop_type or 'receptions' in prop_type:
            # Pass defense (same as passing)
            defense_stats = pbp_filtered.filter(
                (pl.col('defteam') == opponent_team) &
                (pl.col('pass_attempt') == 1)
            ).group_by('defteam').agg([
                pl.col('passing_yards').sum().alias('yards_allowed'),
                pl.col('pass_attempt').sum().alias('attempts_faced')
            ])

            if len(defense_stats) == 0:
                return 1.0

            yards_per_attempt = defense_stats['yards_allowed'][0] / max(defense_stats['attempts_faced'][0], 1)

            if yards_per_attempt < 6.0:
                return 0.85
            elif yards_per_attempt < 6.5:
                return 0.90
            elif yards_per_attempt < 7.5:
                return 1.0
            elif yards_per_attempt < 8.0:
                return 1.10
            else:
                return 1.15

        return 1.0

    except Exception as e:
        logger.warning(f"Error calculating opponent defense adjustment: {e}")
        return 1.0


def apply_weather_adjustment(
    player_id: str,
    season: int,
    position: str,
    prop_type: str,
    game_weather: Optional[Dict]
) -> float:
    """
    Apply weather adjustment for game conditions.

    Integrates with existing weather system from weather_cache_builder.py

    Args:
        game_weather: Dict with 'temp', 'wind', 'weather', 'roof'

    Returns:
        Multiplier based on weather impact for this position/prop type
    """
    if not game_weather:
        return 1.0

    try:
        from modules.weather_cache_builder import calculate_weather_adjustment

        # Extract weather fields
        temp = game_weather.get('temp', 70.0)
        wind = game_weather.get('wind', 0.0)
        weather = game_weather.get('weather', 'clear')
        roof = game_weather.get('roof', 'outdoors')

        # Call existing weather adjustment function
        multiplier = calculate_weather_adjustment(
            player_id=player_id,
            season=season,
            position=position,
            game_temp=temp,
            game_wind=wind,
            game_weather=weather,
            game_roof=roof
        )

        logger.debug(f"Weather adjustment for {player_id}: {multiplier:.3f}x (temp={temp}, wind={wind})")
        return multiplier

    except Exception as e:
        logger.warning(f"Error calculating weather adjustment: {e}")
        return 1.0


def apply_success_rate_adjustment(
    player_id: str,
    season: int,
    week: int,
    through_week: int,
    position: str,
    prop_type: str
) -> float:
    """
    Apply success rate adjustment based on player's 3-week rolling efficiency.

    Success rate measures chain-moving plays:
    - Pass: 1st down gained OR gained 40%+ of yards to go
    - Rush: 1st down gained OR gained 40%+ of yards to go

    Args:
        player_id: Player GSIS ID
        through_week: Calculate through this week (3-week rolling from through_week-2 to through_week)

    Returns:
        Multiplier based on success rate tier:
        - >70%: 1.15x (Critical)
        - 60-70%: 1.10x (High)
        - 50-60%: 1.05x (Average)
        - <50%: 0.92x (Low)
    """
    try:
        # Load play-by-play data
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            return 1.0

        pbp = pl.read_parquet(pbp_file)

        # Filter to last 3 weeks through through_week
        start_week = max(1, through_week - 2)
        pbp_filtered = pbp.filter(
            (pl.col('week') >= start_week) &
            (pl.col('week') <= through_week)
        )

        # Calculate success rate based on position and prop type
        if position == 'QB' and 'passing' in prop_type:
            # QB passing success rate
            player_plays = pbp_filtered.filter(
                (pl.col('passer_player_id') == player_id) &
                (pl.col('pass_attempt') == 1)
            )

            if len(player_plays) < 20:  # Minimum sample size
                return 1.0

            # Success: first down OR gained 40%+ of yards to go
            successful_plays = player_plays.filter(
                (pl.col('first_down_pass') == 1) |
                ((pl.col('yards_gained') / pl.col('ydstogo')).fill_null(0) >= 0.4)
            )

            success_rate = len(successful_plays) / len(player_plays)

        elif 'rushing' in prop_type:
            # Rushing success rate
            player_plays = pbp_filtered.filter(
                (pl.col('rusher_player_id') == player_id) &
                (pl.col('rush_attempt') == 1)
            )

            if len(player_plays) < 20:
                return 1.0

            successful_plays = player_plays.filter(
                (pl.col('first_down_rush') == 1) |
                ((pl.col('yards_gained') / pl.col('ydstogo')).fill_null(0) >= 0.4)
            )

            success_rate = len(successful_plays) / len(player_plays)

        elif 'receiving' in prop_type or 'receptions' in prop_type:
            # Receiver success rate (successful targets)
            player_plays = pbp_filtered.filter(
                (pl.col('receiver_player_id') == player_id) &
                ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
            )

            if len(player_plays) < 15:
                return 1.0

            successful_plays = player_plays.filter(
                (pl.col('complete_pass') == 1) &
                ((pl.col('first_down_pass') == 1) |
                 ((pl.col('yards_gained') / pl.col('ydstogo')).fill_null(0) >= 0.4))
            )

            success_rate = len(successful_plays) / len(player_plays)

        else:
            return 1.0

        # Tier 2: Moderate ranges ±17.5% (more conservative after baseline improvements)
        # Success rate tiers based on play-level efficiency
        if success_rate > 0.70:
            multiplier = 1.175  # Elite efficiency (>70%)
        elif success_rate > 0.60:
            multiplier = 1.09   # High efficiency (60-70%)
        elif success_rate > 0.50:
            multiplier = 1.04   # Above average (50-60%)
        elif success_rate > 0.40:
            multiplier = 0.96   # Below average (40-50%)
        else:
            multiplier = 0.825  # Low efficiency (<40%)

        logger.debug(f"Success rate for {player_id}: {success_rate:.3f} → {multiplier:.3f}x")
        return multiplier

    except Exception as e:
        logger.warning(f"Error calculating success rate adjustment: {e}")
        return 1.0


def apply_catch_rate_adjustment(
    player_id: str,
    season: int,
    through_week: int,
    position: str
) -> float:
    """
    Apply catch rate adjustment based on target efficiency.

    Compares player's catch rate to league average for their route depth.

    Returns:
        Multiplier: 1.10x (excellent hands) to 0.90x (poor hands)
    """
    try:
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            return 1.0

        pbp = pl.read_parquet(pbp_file)
        pbp_filtered = pbp.filter(pl.col('week') <= through_week)

        # Get player targets
        targets = pbp_filtered.filter(
            (pl.col('receiver_player_id') == player_id) &
            ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
        )

        if len(targets) < 15:
            return 1.0

        completions = targets.filter(pl.col('complete_pass') == 1)
        catch_rate = len(completions) / len(targets)

        # Apply multiplier based on catch rate
        if catch_rate > 0.75:
            return 1.10  # Excellent hands
        elif catch_rate > 0.68:
            return 1.05  # Good hands
        elif catch_rate > 0.60:
            return 1.0   # Average
        elif catch_rate > 0.50:
            return 0.95  # Below average
        else:
            return 0.90  # Poor hands

    except Exception as e:
        logger.warning(f"Error calculating catch rate adjustment: {e}")
        return 1.0


def apply_blocking_quality_adjustment(
    player_id: str,
    season: int,
    through_week: int
) -> float:
    """
    Apply blocking quality adjustment for RBs.

    Compares RB's YPC to teammate RBs as proxy for OL quality.

    Returns:
        Multiplier: 1.10x (good blocking) to 0.90x (poor blocking)
    """
    try:
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            return 1.0

        pbp = pl.read_parquet(pbp_file)
        pbp_filtered = pbp.filter(
            (pl.col('week') <= through_week) &
            (pl.col('rush_attempt') == 1)
        )

        # Get player's carries
        player_carries = pbp_filtered.filter(pl.col('rusher_player_id') == player_id)

        if len(player_carries) < 20:
            return 1.0

        player_ypc = player_carries['yards_gained'].sum() / len(player_carries)

        # Get team's other RBs YPC
        player_team = player_carries['posteam'][0] if len(player_carries) > 0 else None
        if not player_team:
            return 1.0

        team_other_carries = pbp_filtered.filter(
            (pl.col('posteam') == player_team) &
            (pl.col('rusher_player_id') != player_id)
        )

        if len(team_other_carries) < 20:
            return 1.0

        team_ypc = team_other_carries['yards_gained'].sum() / len(team_other_carries)

        # Compare player YPC to team YPC
        if player_ypc > team_ypc * 1.15:
            return 1.10  # Player significantly better (good blocking or talent)
        elif player_ypc > team_ypc * 1.05:
            return 1.05
        elif player_ypc > team_ypc * 0.95:
            return 1.0
        elif player_ypc > team_ypc * 0.85:
            return 0.95
        else:
            return 0.90

    except Exception as e:
        logger.warning(f"Error calculating blocking quality adjustment: {e}")
        return 1.0


def apply_separation_adjustment(
    player_id: str,
    season: int,
    through_week: int,
    position: str
) -> float:
    """
    Apply separation adjustment for receivers (NextGen data).

    Note: NextGen separation data not available in standard nflfastR data.
    Returns neutral multiplier for now.

    Returns:
        1.0 (neutral - awaiting NextGen data integration)
    """
    # NextGen data integration pending
    return 1.0


def apply_route_location_adjustment(
    player_id: str,
    season: int,
    through_week: int,
    position: str
) -> float:
    """
    Apply route location adjustment based on field position tendencies.

    Rewards players who get more red zone targets/carries.

    Returns:
        Multiplier: 1.08x (red zone specialist) to 0.95x (limited red zone)
    """
    try:
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            return 1.0

        pbp = pl.read_parquet(pbp_file)
        pbp_filtered = pbp.filter(pl.col('week') <= through_week)

        # Get player's touches
        if position in ['WR', 'TE']:
            touches = pbp_filtered.filter(
                (pl.col('receiver_player_id') == player_id) &
                (pl.col('complete_pass') == 1)
            )
        elif position == 'RB':
            touches = pbp_filtered.filter(
                (pl.col('rusher_player_id') == player_id) &
                (pl.col('rush_attempt') == 1)
            )
        else:
            return 1.0

        if len(touches) < 15:
            return 1.0

        # Count red zone touches (inside opponent 20)
        red_zone_touches = touches.filter(pl.col('yardline_100') <= 20)
        red_zone_rate = len(red_zone_touches) / len(touches)

        # Apply multiplier
        if red_zone_rate > 0.25:
            return 1.08  # Red zone specialist
        elif red_zone_rate > 0.18:
            return 1.04  # Good red zone usage
        elif red_zone_rate > 0.12:
            return 1.0   # Average
        else:
            return 0.95  # Limited red zone usage

    except Exception as e:
        logger.warning(f"Error calculating route location adjustment: {e}")
        return 1.0


def apply_penalty_adjustment(
    player_id: str,
    season: int,
    week: int,
    through_week: int
) -> float:
    """
    Apply penalty adjustment for frequently penalized players.

    Penalizes players with high penalty rates.

    Returns:
        Multiplier: 0.92x (high penalties) to 1.0x (clean player)
    """
    try:
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            return 1.0

        pbp = pl.read_parquet(pbp_file)
        pbp_filtered = pbp.filter(pl.col('week') <= through_week)

        # Get player's touches/plays
        player_plays = pbp_filtered.filter(
            (pl.col('passer_player_id') == player_id) |
            (pl.col('rusher_player_id') == player_id) |
            (pl.col('receiver_player_id') == player_id)
        )

        if len(player_plays) < 20:
            return 1.0

        # Count penalties on player
        penalties = player_plays.filter(pl.col('penalty') == 1)
        penalty_rate = len(penalties) / len(player_plays)

        # Apply penalty
        if penalty_rate > 0.08:
            return 0.92  # High penalty rate
        elif penalty_rate > 0.05:
            return 0.96  # Moderate penalties
        else:
            return 1.0   # Clean player

    except Exception as e:
        logger.warning(f"Error calculating penalty adjustment: {e}")
        return 1.0


def apply_turnover_adjustment(
    player_id: str,
    season: int,
    week: int,
    through_week: int,
    position: str
) -> float:
    """
    Apply turnover adjustment for turnover-prone players.

    Penalizes QBs/RBs/WRs with high turnover rates.

    Returns:
        Multiplier: 0.90x (turnover-prone) to 1.0x (ball security)
    """
    try:
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            return 1.0

        pbp = pl.read_parquet(pbp_file)
        pbp_filtered = pbp.filter(pl.col('week') <= through_week)

        # Count turnovers
        if position == 'QB':
            plays = pbp_filtered.filter(
                (pl.col('passer_player_id') == player_id) &
                (pl.col('pass_attempt') == 1)
            )
            turnovers = plays.filter(pl.col('interception') == 1)

        elif position == 'RB':
            plays = pbp_filtered.filter(
                (pl.col('rusher_player_id') == player_id) &
                (pl.col('rush_attempt') == 1)
            )
            turnovers = plays.filter(pl.col('fumble_lost') == 1)

        else:
            # Receivers - fumbles only
            plays = pbp_filtered.filter(
                (pl.col('receiver_player_id') == player_id) &
                (pl.col('complete_pass') == 1)
            )
            turnovers = plays.filter(pl.col('fumble_lost') == 1)

        if len(plays) < 30:
            return 1.0

        turnover_rate = len(turnovers) / len(plays)

        # Apply penalty
        if turnover_rate > 0.04:
            return 0.90  # Very turnover-prone
        elif turnover_rate > 0.025:
            return 0.95  # Moderate turnovers
        else:
            return 1.0   # Good ball security

    except Exception as e:
        logger.warning(f"Error calculating turnover adjustment: {e}")
        return 1.0


if __name__ == "__main__":
    # Test projection generation
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("=== Testing prop_projection_engine.py ===")
    print("Module loaded successfully")
    print("\nTo test:")
    print("1. Get player GSIS ID from cache/rosters/rosters-2024.csv")
    print("2. Call generate_projection(player_id, 2024, 10, 'QB', 'passing_yards')")
    print("3. Verify baseline calculation and adjustment application")
    print("\nNote: Adjustment functions are placeholders (returning 1.0)")
    print("Will integrate with existing adjustment pipeline next")
