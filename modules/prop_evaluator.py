"""
Player Props Evaluator

Compares GridironMetrics projections against betting lines to identify value bets.

Value Identification Logic:
- Calculate edge: (projection - line) / line
- Minimum edge threshold: 8% (configurable)
- Consider both over and under opportunities
- Flag high-confidence bets based on variance (CV)

Output:
- Ranked value bets by edge %
- Confidence ratings (A/B/C based on variance and sample size)
"""

import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from modules.logger import get_logger
from modules.prop_projection_engine import generate_projection
from modules.prop_types import get_prop_types_for_position, get_display_name, get_api_market_for_prop

logger = get_logger(__name__)


class PropEvaluator:
    """Evaluates projections vs betting lines to identify value."""

    def __init__(self, min_edge: float = 0.08):
        """
        Initialize evaluator.

        Args:
            min_edge: Minimum edge percentage to flag as value (default 8%)
        """
        self.min_edge = min_edge

    def calculate_confidence_grade(
        self,
        variance: float,
        games_played: int,
        edge: float
    ) -> str:
        """
        Calculate confidence grade for a bet.

        Args:
            variance: Coefficient of variation (CV)
            games_played: Number of games in sample
            edge: Edge percentage

        Returns:
            'A' (high confidence), 'B' (medium), or 'C' (low)

        Grading Logic:
        - A: Low variance (<0.20) + good sample (8+ games) + strong edge (>12%)
        - B: Medium variance (<0.30) + decent sample (5+ games) + moderate edge (>8%)
        - C: Everything else that meets minimum edge
        """
        if variance < 0.20 and games_played >= 8 and edge > 0.12:
            return 'A'
        elif variance < 0.30 and games_played >= 5 and edge > 0.08:
            return 'B'
        else:
            return 'C'

    def evaluate_prop(
        self,
        player_id: str,
        player_name: str,
        season: int,
        week: int,
        position: str,
        prop_type: str,
        betting_line: float,
        opponent_team: Optional[str] = None,
        game_weather: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Evaluate a single prop bet.

        Args:
            player_id: Player GSIS ID
            player_name: Player display name
            season: Season year
            week: Week number
            position: Position
            prop_type: Prop type (e.g., 'passing_yards')
            betting_line: Sportsbook line
            opponent_team: Opponent abbreviation
            game_weather: Weather dict

        Returns:
            Dict with evaluation results or None if no edge
            {
                'player_id': str,
                'player_name': str,
                'prop_type': str,
                'betting_line': float,
                'projection': float,
                'edge': float,
                'edge_pct': float,
                'confidence': str (A/B/C),
                'recommendation': str (OVER/UNDER),
                'variance': float,
                'games_played': int,
                'adjustments': dict
            }
        """
        # Generate projection
        projection = generate_projection(
            player_id=player_id,
            season=season,
            week=week,
            position=position,
            prop_type=prop_type,
            opponent_team=opponent_team,
            game_weather=game_weather
        )

        if not projection or projection['final_projection'] == 0:
            logger.warning(f"No projection available for {player_name} {prop_type}")
            return None

        proj_value = projection['final_projection']
        variance = projection['stat_summary']['variance']
        games_played = projection['stat_summary']['games_played']

        # Calculate edge
        edge = proj_value - betting_line
        edge_pct = edge / betting_line if betting_line != 0 else 0

        # Determine if this is value
        if abs(edge_pct) < self.min_edge:
            return None  # No edge

        # Determine recommendation
        recommendation = "OVER" if edge > 0 else "UNDER"

        # Calculate confidence
        confidence = self.calculate_confidence_grade(
            variance=variance,
            games_played=games_played,
            edge=abs(edge_pct)
        )

        return {
            'player_id': player_id,
            'player_name': player_name,
            'position': position,
            'prop_type': prop_type,
            'prop_display': get_display_name(prop_type),
            'betting_line': betting_line,
            'projection': proj_value,
            'edge': edge,
            'edge_pct': edge_pct,
            'confidence': confidence,
            'recommendation': recommendation,
            'variance': variance,
            'games_played': games_played,
            'effective_games': projection['effective_games'],
            'adjustments': projection['adjustments'],
            'opponent': opponent_team or 'N/A'
        }

    def evaluate_player_props(
        self,
        player_id: str,
        player_name: str,
        season: int,
        week: int,
        position: str,
        betting_lines: Dict[str, float],
        opponent_team: Optional[str] = None,
        game_weather: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Evaluate all prop bets for a player.

        Args:
            player_id: Player GSIS ID
            player_name: Player display name
            season: Season year
            week: Week number
            position: Position
            betting_lines: Dict of {prop_type: line_value}
            opponent_team: Opponent abbreviation
            game_weather: Weather dict

        Returns:
            List of value bets sorted by edge %
        """
        value_bets = []

        for prop_type, line in betting_lines.items():
            evaluation = self.evaluate_prop(
                player_id=player_id,
                player_name=player_name,
                season=season,
                week=week,
                position=position,
                prop_type=prop_type,
                betting_line=line,
                opponent_team=opponent_team,
                game_weather=game_weather
            )

            if evaluation:
                value_bets.append(evaluation)

        # Sort by absolute edge %
        value_bets.sort(key=lambda x: abs(x['edge_pct']), reverse=True)

        return value_bets

    def evaluate_week(
        self,
        season: int,
        week: int,
        betting_lines_data: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        """
        Evaluate all props for a week.

        Args:
            season: Season year
            week: Week number
            betting_lines_data: List of dicts with player betting lines:
                [
                    {
                        'player_id': str,
                        'player_name': str,
                        'position': str,
                        'opponent_team': str,
                        'lines': {prop_type: line_value},
                        'weather': {temp, wind, weather, roof} (optional)
                    },
                    ...
                ]

        Returns:
            Tuple of (value_bets, summary)
            - value_bets: List of all value bets sorted by edge %
            - summary: Dict with statistics
        """
        all_value_bets = []
        total_props_evaluated = 0
        total_value_found = 0

        for player_data in betting_lines_data:
            player_value_bets = self.evaluate_player_props(
                player_id=player_data['player_id'],
                player_name=player_data['player_name'],
                season=season,
                week=week,
                position=player_data['position'],
                betting_lines=player_data['lines'],
                opponent_team=player_data.get('opponent_team'),
                game_weather=player_data.get('weather')
            )

            all_value_bets.extend(player_value_bets)
            total_props_evaluated += len(player_data['lines'])
            total_value_found += len(player_value_bets)

        # Sort all value bets by edge %
        all_value_bets.sort(key=lambda x: abs(x['edge_pct']), reverse=True)

        # Generate summary statistics
        summary = {
            'total_props_evaluated': total_props_evaluated,
            'total_value_found': total_value_found,
            'value_pct': (total_value_found / total_props_evaluated * 100) if total_props_evaluated > 0 else 0,
            'confidence_breakdown': {
                'A': len([b for b in all_value_bets if b['confidence'] == 'A']),
                'B': len([b for b in all_value_bets if b['confidence'] == 'B']),
                'C': len([b for b in all_value_bets if b['confidence'] == 'C'])
            },
            'recommendation_breakdown': {
                'OVER': len([b for b in all_value_bets if b['recommendation'] == 'OVER']),
                'UNDER': len([b for b in all_value_bets if b['recommendation'] == 'UNDER'])
            },
            'avg_edge_pct': sum(abs(b['edge_pct']) for b in all_value_bets) / len(all_value_bets) * 100 if all_value_bets else 0
        }

        logger.info(
            f"Week {week} evaluation complete: "
            f"{total_value_found}/{total_props_evaluated} value bets found "
            f"({summary['value_pct']:.1f}%), avg edge {summary['avg_edge_pct']:.1f}%"
        )

        return all_value_bets, summary


if __name__ == "__main__":
    # Test evaluator
    print("=== Testing prop_evaluator.py ===")
    print("Module loaded successfully")
    print("\nEvaluator Configuration:")
    evaluator = PropEvaluator(min_edge=0.08)
    print(f"  Minimum edge: {evaluator.min_edge * 100}%")
    print("\nTo test:")
    print("1. Provide betting lines data in format:")
    print("   [{")
    print("     'player_id': '00-0033873',")
    print("     'player_name': 'Patrick Mahomes',")
    print("     'position': 'QB',")
    print("     'opponent_team': 'BUF',")
    print("     'lines': {'passing_yards': 275.5, 'passing_tds': 2.5},")
    print("     'weather': {'temp': 55, 'wind': 10, 'weather': 'cloudy', 'roof': 'outdoors'}")
    print("   }]")
    print("2. Call evaluator.evaluate_week(2024, 11, betting_lines_data)")
    print("3. Review value bets with edge > 8%")
