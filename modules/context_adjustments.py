"""
context_adjustments.py

Advanced contextual adjustments for player performance metrics.
Includes third down performance, garbage time detection, catch rate analysis,
blocking quality proxies, and talent adjustment calculations.
"""

import polars as pl
from typing import Dict, Tuple, Optional
from .logger import get_logger

logger = get_logger(__name__)


class ContextAdjustments:
    """
    Handles player-level contextual adjustments that go beyond basic situational multipliers.
    These adjustments account for difficulty, teammate quality, and specialized situations.
    """
    
    def __init__(self):
        self.logger = logger
    
    # ============================================================================
    # THIRD DOWN PERFORMANCE MULTIPLIERS
    # ============================================================================
    
    def calculate_third_down_multiplier(self, down: int, ydstogo: int) -> float:
        """
        Calculate multiplier for third down plays based on distance.
        
        Rationale: Converting 3rd & 10 is more valuable than 3rd & 1.
        
        Args:
            down: Down number (1-4)
            ydstogo: Yards to go for first down
            
        Returns:
            float: Multiplier between 1.0 (not 3rd down) and 1.6 (3rd & 15+)
            
        Multiplier Scale:
            - Not 3rd down: 1.0x (no adjustment)
            - 3rd & 1: 1.05x (short distance)
            - 3rd & 2-3: 1.15x (medium-short)
            - 3rd & 4-6: 1.30x (medium)
            - 3rd & 7-9: 1.45x (medium-long)
            - 3rd & 10-14: 1.55x (long)
            - 3rd & 15+: 1.6x (very long)
        """
        if down != 3:
            return 1.0
        
        if ydstogo == 1:
            return 1.05
        elif ydstogo <= 3:
            return 1.15
        elif ydstogo <= 6:
            return 1.30
        elif ydstogo <= 9:
            return 1.45
        elif ydstogo <= 14:
            return 1.55
        else:  # 15+
            return 1.60
    
    # ============================================================================
    # GARBAGE TIME DETECTION
    # ============================================================================
    
    def calculate_garbage_time_multiplier(
        self, 
        score_differential: int, 
        game_seconds_remaining: int
    ) -> float:
        """
        Calculate penalty multiplier for garbage time situations.
        
        Penalizes stats accumulated when the losing team is down big with little time left.
        Philosophy: These stats have less value because the defense is playing prevent
        and the game outcome is essentially decided.
        
        Args:
            score_differential: Point differential (positive = winning, negative = losing)
            game_seconds_remaining: Seconds remaining in game
            
        Returns:
            float: Penalty multiplier between 0.6x (severe garbage time) and 1.0x (competitive)
            
        Garbage Time Criteria:
            - Score differential > 17 points (3-score game)
            - Time remaining <= 8 minutes (480 seconds)
            - Only applies to LOSING team (score_differential < 0)
            
        Penalty Scale:
            - Down 18-23 pts, 8 min left: ~0.9x (mild penalty, still somewhat competitive)
            - Down 24+ pts, 5 min left: ~0.72x (moderate penalty)
            - Down 24+ pts, 2 min left: ~0.65x (severe penalty, pure garbage time)
            - Down 24+ pts, 0 min left: ~0.6x (maximum penalty)
        """
        # Not garbage time if score differential is 17 or less
        if abs(score_differential) <= 17:
            return 1.0
        
        # Not garbage time if more than 8 minutes remaining
        if game_seconds_remaining > 480:
            return 1.0
        
        # Only penalize the LOSING team
        if score_differential >= 0:  # Winning or tied
            return 1.0
        
        # Calculate time factor (0.0 to 1.0, where 0.0 = no time left, 1.0 = 8 min left)
        time_factor = max(0.0, game_seconds_remaining / 480.0)
        
        # Calculate penalty: 0.6 base + up to 0.3 based on time remaining
        # More time = less penalty (closer to 0.9)
        # Less time = more penalty (closer to 0.6)
        penalty_multiplier = 0.6 + (0.3 * time_factor)
        
        return penalty_multiplier
    
    # ============================================================================
    # YAC MULTIPLIER FOR RECEIVERS
    # ============================================================================
    
    def calculate_yac_multiplier(
        self,
        air_yards: Optional[float],
        yards_after_catch: Optional[float],
        yards_gained: float
    ) -> float:
        """
        Calculate multiplier based on yards after catch percentage.
        
        Rewards receivers who create their own yards after the catch vs. those who
        rely purely on QB accuracy and scheme. High YAC% indicates elusiveness,
        tackle-breaking ability, and yards creation.
        
        Args:
            air_yards: Yards ball traveled in air (not used currently, for future depth adjustment)
            yards_after_catch: Yards gained after catch
            yards_gained: Total receiving yards on play
            
        Returns:
            float: Multiplier between 0.95x (low YAC) and 1.15x (high YAC)
            
        YAC Percentage Tiers:
            - 70%+ YAC: 1.15x (elite yards creator - screen master or tackle breaker)
            - 50-70% YAC: 1.10x (good YAC ability)
            - 30-50% YAC: 1.05x (solid balance)
            - 10-30% YAC: 1.0x (neutral - average)
            - <10% YAC: 0.95x (pure air yards - contested catch or QB placement)
            
        Philosophy:
            - High YAC% = receiver creating value through elusiveness
            - Low YAC% = value created by QB accuracy or contested catches
            - We slightly penalize pure air yards to reward playmaking ability
        """
        # Handle missing or invalid data
        if yards_after_catch is None or yards_gained <= 0:
            return 1.0
        
        # Avoid division by zero
        if yards_gained == 0:
            return 1.0
        
        # Calculate YAC percentage
        yac_percentage = yards_after_catch / yards_gained
        
        # Apply tier-based multiplier
        if yac_percentage > 0.7:
            return 1.15  # Elite YAC
        elif yac_percentage > 0.5:
            return 1.10  # Good YAC
        elif yac_percentage > 0.3:
            return 1.05  # Solid YAC
        elif yac_percentage >= 0.1:
            return 1.0   # Average YAC
        else:  # < 0.1
            return 0.95  # Low YAC (pure air yards)
    
    # ============================================================================
    # CATCH RATE ANALYSIS
    # ============================================================================
    
    def calculate_catch_rate_adjustment(
        self,
        pbp_data: pl.DataFrame,
        player_id: str
    ) -> float:
        """
        Calculate catch rate adjustment based on actual vs expected catch rate.
        
        Compares player's catch rate to depth-adjusted expectations. Deep targets
        (air_yards > 20) have lower expected catch rates than short routes.
        
        Args:
            pbp_data: Play-by-play DataFrame (must include complete_pass, incomplete_pass,
                     receiver_player_id, air_yards columns)
            player_id: Player ID to calculate adjustment for
            
        Returns:
            float: Multiplier between 0.90 and 1.10
            
        Multiplier Scale:
            - Catch rate +8% over expected: 1.10x (excellent hands/separation)
            - Catch rate +5% over expected: 1.07x (very good)
            - Catch rate +2% over expected: 1.03x (above average)
            - Catch rate -2% to +2%: 1.0x (league average)
            - Catch rate -2% to -5%: 0.97x (below average)
            - Catch rate -5% to -8%: 0.93x (concerning)
            - Catch rate -8%+ below expected: 0.90x (poor hands/chemistry)
            
        Note: Only applies to WR and TE positions. Requires minimum 20 targets.
        """
        # Filter to player's targets (completions + incompletions)
        player_targets = pbp_data.filter(
            (pl.col("receiver_player_id") == player_id) &
            ((pl.col("complete_pass") == 1) | (pl.col("incomplete_pass") == 1))
        )
        
        # Check minimum sample size (20 targets)
        target_count = len(player_targets)
        if target_count < 20:
            self.logger.debug(f"Player {player_id} has only {target_count} targets (min 20), returning neutral multiplier")
            return 1.0
        
        # Calculate actual catch rate
        completions = player_targets.filter(pl.col("complete_pass") == 1).shape[0]
        actual_catch_rate = completions / target_count
        
        # Calculate average target depth
        avg_depth = player_targets.select(pl.col("air_yards").mean()).item()
        
        # Determine expected catch rate based on depth
        if avg_depth >= 20:
            expected_catch_rate = 0.45  # Deep ball
        elif avg_depth >= 15:
            expected_catch_rate = 0.55  # Intermediate-deep
        elif avg_depth >= 10:
            expected_catch_rate = 0.65  # Intermediate
        else:
            expected_catch_rate = 0.75  # Short routes
        
        # Calculate catch rate over expected
        catch_rate_diff = actual_catch_rate - expected_catch_rate
        
        # Apply tiered multiplier
        if catch_rate_diff >= 0.08:
            multiplier = 1.10
        elif catch_rate_diff >= 0.05:
            multiplier = 1.07
        elif catch_rate_diff >= 0.02:
            multiplier = 1.03
        elif catch_rate_diff >= -0.02:
            multiplier = 1.0
        elif catch_rate_diff >= -0.05:
            multiplier = 0.97
        elif catch_rate_diff >= -0.08:
            multiplier = 0.93
        else:
            multiplier = 0.90
        
        self.logger.debug(
            f"Player {player_id}: {target_count} targets, "
            f"{actual_catch_rate:.1%} catch rate (exp: {expected_catch_rate:.1%}, "
            f"diff: {catch_rate_diff:+.1%}), "
            f"avg depth: {avg_depth:.1f}yd → {multiplier}x"
        )
        
        return multiplier
    
    # ============================================================================
    # BLOCKING QUALITY PROXY
    # ============================================================================
    
    def calculate_blocking_quality_proxy(
        self,
        pbp_data: pl.DataFrame,
        player_id: str,
        player_team: str
    ) -> float:
        """
        Estimate blocking quality by comparing player YPC to teammate RB average.
        
        This is a proxy for offensive line quality. If a player significantly
        outperforms their teammates, they are likely creating their own yards.
        If they underperform, the OL may be creating opportunities.
        
        Args:
            pbp_data: Play-by-play DataFrame (must include rusher_player_id, yards_gained)
            player_id: Player ID to calculate adjustment for
            player_team: Team abbreviation
            
        Returns:
            float: Multiplier between 0.95 and 1.05
            
        Multiplier Scale:
            - Player YPC 1.3x+ teammate avg: 1.05x (creating own yards)
            - Player YPC 1.15-1.3x teammate avg: 1.03x (above average)
            - Player YPC 0.85-1.15x teammate avg: 1.0x (neutral)
            - Player YPC 0.7-0.85x teammate avg: 0.98x (below average)
            - Player YPC <0.7x teammate avg: 0.95x (dependent on blocking)
            
        Note: Requires minimum 20 carries for player and 30 carries for teammates.
        """
        # Get player's rushing stats
        player_rushes = pbp_data.filter(
            (pl.col("rusher_player_id") == player_id) &
            (pl.col("rush_attempt") == 1)
        )
        
        player_carries = len(player_rushes)
        if player_carries < 20:
            self.logger.debug(f"Player {player_id} has only {player_carries} carries (min 20), returning neutral multiplier")
            return 1.0
        
        player_yards = player_rushes.select(pl.col("yards_gained").sum()).item()
        player_ypc = player_yards / player_carries
        
        # Get teammate RB stats (same team, different player)
        teammate_rushes = pbp_data.filter(
            (pl.col("rusher_player_id") != player_id) &
            (pl.col("posteam") == player_team) &
            (pl.col("rush_attempt") == 1)
        )
        
        teammate_carries = len(teammate_rushes)
        if teammate_carries < 30:
            self.logger.debug(f"Team {player_team} teammates have only {teammate_carries} carries (min 30), returning neutral multiplier")
            return 1.0
        
        teammate_yards = teammate_rushes.select(pl.col("yards_gained").sum()).item()
        teammate_ypc = teammate_yards / teammate_carries
        
        # Avoid division by zero
        if teammate_ypc == 0:
            return 1.0
        
        # Calculate ratio
        ypc_ratio = player_ypc / teammate_ypc
        
        # Apply tiered multiplier
        if ypc_ratio >= 1.3:
            multiplier = 1.05
        elif ypc_ratio >= 1.15:
            multiplier = 1.03
        elif ypc_ratio >= 0.85:
            multiplier = 1.0
        elif ypc_ratio >= 0.7:
            multiplier = 0.98
        else:
            multiplier = 0.95
        
        self.logger.debug(
            f"Player {player_id}: {player_ypc:.2f} YPC vs teammates {teammate_ypc:.2f} YPC "
            f"(ratio: {ypc_ratio:.2f}) → {multiplier}x"
        )
        
        return multiplier
    
    # ============================================================================
    # TALENT ADJUSTMENT
    # ============================================================================
    
    def calculate_teammate_quality_index(
        self,
        player_id: str,
        player_name: str,
        player_team: str,
        player_position: str,
        all_players_scores: pl.DataFrame
    ) -> float:
        """
        Calculate talent adjustment based on supporting cast quality.
        
        Uses a two-pass system:
        1. First pass: Calculate baseline scores for all players (done externally)
        2. Second pass: For each player, sum their teammates' scores and convert to percentile
        3. Apply inverse adjustment: boost players with weak teammates, penalize those with elite support
        
        Args:
            player_id: Player ID to calculate adjustment for
            player_name: Player name
            player_team: Team abbreviation
            player_position: Player position (RB/WR/TE)
            all_players_scores: DataFrame with columns [player_id, player_name, team, position, baseline_score]
            
        Returns:
            float: Multiplier between 0.95 and 1.08
            
        Multiplier Scale:
            - Teammate quality 0-25th percentile (weak): 1.08x (carrying weak offense)
            - Teammate quality 25-40th percentile: 1.05x (above average difficulty)
            - Teammate quality 40-60th percentile: 1.0x (neutral)
            - Teammate quality 60-75th percentile: 0.97x (slight benefit from talent)
            - Teammate quality 75-100th percentile (elite): 0.95x (surrounded by stars)
            
        Note: Teammates defined as other RB/WR/TE on same team (excludes self).
        """
        # Get all skill position players on the same team (excluding self)
        teammates = all_players_scores.filter(
            (pl.col('team') == player_team) &
            (pl.col('position').is_in(['RB', 'WR', 'TE'])) &
            ~((pl.col('player_id') == player_id) & (pl.col('player_name') == player_name))
        )
        
        # Check if player has teammates
        if len(teammates) == 0:
            self.logger.debug(f"Player {player_name} has no teammates on {player_team}, returning neutral multiplier")
            return 1.0
        
        # Sum teammate baseline scores
        teammate_total_score = teammates.select(pl.col('baseline_score').sum()).item()
        
        # Calculate percentile rank of teammate quality across all teams
        # Higher percentile = better teammates
        all_team_scores = []
        for team in all_players_scores['team'].unique():
            team_players = all_players_scores.filter(
                (pl.col('team') == team) &
                (pl.col('position').is_in(['RB', 'WR', 'TE']))
            )
            if len(team_players) > 0:
                team_score = team_players.select(pl.col('baseline_score').sum()).item()
                all_team_scores.append(team_score)
        
        if len(all_team_scores) == 0:
            return 1.0
        
        # Calculate percentile (0-100)
        percentile = sum(1 for s in all_team_scores if s < teammate_total_score) / len(all_team_scores) * 100
        
        # Apply tiered multiplier (inverse - low percentile = weak teammates = boost)
        if percentile < 25:
            multiplier = 1.08  # Weakest 25% of supporting casts
        elif percentile < 40:
            multiplier = 1.05
        elif percentile < 60:
            multiplier = 1.0  # Average supporting cast
        elif percentile < 75:
            multiplier = 0.97
        else:
            multiplier = 0.95  # Top 25% of supporting casts
        
        self.logger.debug(
            f"Player {player_name} ({player_team}): Teammate score {teammate_total_score:.1f}, "
            f"percentile {percentile:.1f}% → {multiplier}x"
        )
        
        return multiplier
    
    # ============================================================================
    # SAMPLE SIZE DAMPENING
    # ============================================================================
    
    def apply_sample_size_dampening(
        self,
        score: float,
        games_played: int,
        full_season_games: int = 17
    ) -> float:
        """
        Apply 0.4 root curve dampening for small sample sizes.
        
        Reduces the impact of small-sample outliers by applying a power curve.
        The 0.4 exponent balances between:
        - Not over-penalizing injury-shortened seasons
        - Properly dampening hot streaks from limited action
        
        Args:
            score: Player's raw score
            games_played: Number of games played
            full_season_games: Games in a full season (default 17)
            
        Returns:
            float: Sample-size adjusted score
            
        Dampening Effect:
            - 17 games: 1.0x (full credit)
            - 10 games: 0.75x credit
            - 8 games: 0.69x credit
            - 5 games: 0.57x credit
            - 2 games: 0.39x credit
            - 1 game: 0.29x credit
        """
        if games_played <= 0:
            return 0.0
        
        if games_played >= full_season_games:
            return score
        
        # Calculate dampening factor using 0.4 root curve
        dampening_factor = (games_played ** 0.4) / (full_season_games ** 0.4)
        
        dampened_score = score * dampening_factor
        
        self.logger.debug(
            f"Sample size dampening: {games_played} games, "
            f"factor {dampening_factor:.3f}, "
            f"score {score:.2f} → {dampened_score:.2f}"
        )
        
        return dampened_score
