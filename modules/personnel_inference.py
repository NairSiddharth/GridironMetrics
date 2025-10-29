"""
personnel_inference.py

Infers offensive personnel groupings (10, 11, 12, 13, 21, 22) from play characteristics
using a multi-factor voting system. Provides confidence scores and position-specific
multipliers to reward/penalize players based on formation context.

Personnel Definitions:
- 10: 1 RB, 0 TE, 4 WR (spread/empty)
- 11: 1 RB, 1 TE, 3 WR (base/standard)
- 12: 1 RB, 2 TE, 2 WR (tight/compressed)
- 13: 1 RB, 3 TE, 1 WR (heavy/goal line)
- 21: 2 RB, 1 TE, 2 WR (power/I-formation)
- 22: 2 RB, 2 TE, 1 WR (jumbo/goal line)
"""

import polars as pl
from typing import Dict, Tuple, List
from .logger import get_logger

logger = get_logger(__name__)


class PersonnelInference:
    """
    Infers offensive personnel groupings using multi-factor voting algorithm.
    """
    
    def __init__(self):
        self.logger = logger
        # Position-specific multipliers for each personnel grouping
        self.wr_multipliers = {
            '10': 1.15,  # 4 WR = defense spread, easier separation
            '11': 1.00,  # Base formation = neutral
            '12': 0.90,  # 2 TE = compressed, harder separation
            '13': 0.85,  # 3 TE = heavy, WR not focal
            '21': 0.90,  # 2 RB = run-focused
            '22': 0.85   # Jumbo = obvious run
        }
        
        self.rb_multipliers = {
            '10': 0.90,  # Spread = likely passing down
            '11': 1.00,  # Base = neutral
            '12': 1.05,  # Tight = better blocking
            '13': 1.08,  # 3 TE = power running lanes
            '21': 1.10,  # 2 RB = heavy, proves skill if successful
            '22': 1.10   # Jumbo = stacked box, elite if successful
        }
        
        self.te_multipliers = {
            '10': 0.85,  # 4 WR = TE not involved
            '11': 1.00,  # Base = neutral
            '12': 1.10,  # 2 TE = TE-focused offense
            '13': 1.10,  # 3 TE = multiple receiving options
            '21': 0.95,  # 2 RB = more blocking than receiving
            '22': 0.95   # Jumbo = blocking role
        }
    
    def infer_personnel(
        self, 
        play_type: str,  # 'pass', 'run', or 'other'
        down: int,
        ydstogo: int,
        yardline_100: int,
        score_differential: int,
        game_seconds_remaining: int,
        receiver_position: str = None,  # For pass plays: 'WR', 'TE', 'RB'
        air_yards: float = None  # For pass plays
    ) -> Tuple[str, float]:
        """
        Infer personnel grouping using multi-factor voting.
        
        Returns:
            Tuple of (personnel_group, confidence_score)
            - personnel_group: '10', '11', '12', '13', '21', or '22'
            - confidence_score: 0.0-1.0 (only apply multiplier if >0.6)
        """
        votes: List[Tuple[str, float]] = []
        
        # VOTE 1: Play Type + Receiver Position (High Confidence)
        if play_type == 'pass':
            if air_yards and air_yards > 20:
                # Deep shot = spread formation
                votes.append(('10', 0.75))
            elif receiver_position == 'TE' and air_yards and air_yards < 10:
                # TE short route = tight formation
                votes.append(('12', 0.70))
            elif receiver_position == 'WR' and air_yards and air_yards > 10:
                # WR intermediate = standard spread
                votes.append(('11', 0.80))
            elif receiver_position == 'RB':
                # RB receiving = likely base
                votes.append(('11', 0.60))
            else:
                # Default pass = base
                votes.append(('11', 0.50))
        
        elif play_type == 'run':
            if down >= 3 and ydstogo <= 2:
                # Short yardage = heavy
                votes.append(('22', 0.70))
            elif yardline_100 <= 5:
                # Goal line = power
                votes.append(('22', 0.65))
            else:
                # Standard rush = base or heavy
                votes.append(('11', 0.55))
        
        # VOTE 2: Down & Distance Context (Medium-High Confidence)
        if down == 3 and ydstogo >= 10:
            # 3rd & long = spread
            votes.append(('10', 0.60))
        elif down == 3 and ydstogo <= 2:
            # 3rd & short = heavy
            votes.append(('22', 0.65))
        elif down == 1 and ydstogo == 10:
            # 1st & 10 = base (very confident)
            votes.append(('11', 0.85))
        elif down == 2 and ydstogo <= 3:
            # 2nd & short = could go heavy
            votes.append(('21', 0.50))
        elif down == 4:
            # 4th down = situation dependent
            if ydstogo <= 2:
                votes.append(('22', 0.60))
            else:
                votes.append(('10', 0.55))
        
        # VOTE 3: Game Script (Medium Confidence)
        if abs(score_differential) > 14 and game_seconds_remaining < 600:
            if score_differential < 0:
                # Losing team = spread to score quickly
                votes.append(('10', 0.65))
            else:
                # Winning team = heavy to run clock
                votes.append(('21', 0.60))
        
        # VOTE 4: Field Position (Medium Confidence)
        if yardline_100 <= 3:
            # Goal line = jumbo
            votes.append(('22', 0.55))
        elif yardline_100 <= 10:
            # Redzone = mix of tight and heavy
            votes.append(('12', 0.45))
        elif yardline_100 >= 80:
            # Own territory = conservative base
            votes.append(('11', 0.50))
        
        # VOTE 5: Time Remaining Context (Lower Confidence)
        if game_seconds_remaining <= 120:
            # Two-minute drill
            if score_differential < 0 or abs(score_differential) <= 7:
                # Need to score = spread
                votes.append(('10', 0.50))
        elif game_seconds_remaining >= 3300:
            # Early game = more balanced
            votes.append(('11', 0.40))
        
        # AGGREGATE VOTES (weighted by confidence)
        if not votes:
            # No votes = default to base with low confidence
            return '11', 0.40
        
        personnel_scores: Dict[str, float] = {}
        for personnel, confidence in votes:
            if personnel not in personnel_scores:
                personnel_scores[personnel] = 0.0
            personnel_scores[personnel] += confidence
        
        # Select highest scoring personnel
        best_personnel = max(personnel_scores, key=personnel_scores.get)
        
        # Calculate aggregate confidence (average of votes for winner)
        winner_votes = [conf for pers, conf in votes if pers == best_personnel]
        avg_confidence = sum(winner_votes) / len(votes) if votes else 0.5
        
        # Normalize confidence to 0.0-1.0 range
        normalized_confidence = min(1.0, avg_confidence)
        
        return best_personnel, normalized_confidence
    
    def get_position_multiplier(
        self, 
        personnel: str, 
        position: str, 
        confidence: float
    ) -> float:
        """
        Get position-specific multiplier for a personnel grouping.
        Only applies if confidence >= 0.6
        
        Args:
            personnel: '10', '11', '12', '13', '21', or '22'
            position: 'WR', 'RB', or 'TE'
            confidence: 0.0-1.0 confidence score
            
        Returns:
            float: Multiplier (0.85-1.15 range, or 1.0 if low confidence)
        """
        # Don't apply if confidence too low
        if confidence < 0.6:
            return 1.0
        
        # Get appropriate multiplier table
        if position == 'WR':
            multipliers = self.wr_multipliers
        elif position == 'RB':
            multipliers = self.rb_multipliers
        elif position == 'TE':
            multipliers = self.te_multipliers
        else:
            return 1.0  # Unknown position
        
        return multipliers.get(personnel, 1.0)
    
    def infer_from_pbp_row(self, row: dict) -> Tuple[str, float]:
        """
        Convenience method to infer personnel from a PBP data row.
        
        Args:
            row: Dictionary with PBP columns
            
        Returns:
            Tuple of (personnel, confidence)
        """
        # Determine play type
        if row.get('pass_attempt', 0) == 1:
            play_type = 'pass'
        elif row.get('rush_attempt', 0) == 1:
            play_type = 'run'
        else:
            play_type = 'other'
        
        # Get receiver position for pass plays
        receiver_pos = None
        if play_type == 'pass':
            # Try to determine from receiver_player_id (would need position lookup)
            # For now, use pass location as proxy
            pass_location = row.get('pass_location', '')
            if 'deep' in pass_location.lower():
                receiver_pos = 'WR'  # Deep passes usually to WRs
            # This is simplified - real implementation would look up actual position
        
        return self.infer_personnel(
            play_type=play_type,
            down=row.get('down', 1),
            ydstogo=row.get('ydstogo', 10),
            yardline_100=row.get('yardline_100', 50),
            score_differential=row.get('score_differential', 0),
            game_seconds_remaining=row.get('game_seconds_remaining', 3600),
            receiver_position=receiver_pos,
            air_yards=row.get('air_yards', None)
        )
