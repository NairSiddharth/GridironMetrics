"""
Handles calculation of offensive metrics, situational adjustments, and opponent strength.
"""

import polars as pl
from pathlib import Path
from typing import Dict, List, Optional
from .logger import get_logger
from .constants import CACHE_DIR

logger = get_logger(__name__)

class OffensiveMetricsCalculator:
    def __init__(self):
        self.defense_rankings: Dict[str, Dict[str, float]] = {}
        
    def calculate_defensive_rankings(self, year: int) -> Dict[str, Dict[str, float]]:
        """
        Calculate defensive rankings for each team based on offensive stats allowed.
        Rankings are normalized on a 0-1 scale where 1 = best defense (allowed least production)
        """
        try:
            # Load all team defensive stats for the year
            team_stats = []
            team_cache_dir = Path(CACHE_DIR) / "team_stats"
            
            for team_dir in team_cache_dir.iterdir():
                if team_dir.is_dir():
                    file_path = team_dir / f"{team_dir.name}-{year}.csv"
                    if file_path.exists():
                        df = pl.read_csv(file_path)
                        team_stats.append(df)
            
            if not team_stats:
                logger.error(f"No team data found for {year}")
                return {}
                
            combined_stats = pl.concat(team_stats)
            
            # Calculate season totals allowed by each defense
            defensive_stats = combined_stats.groupby('opponent').agg([
                pl.col('receiving_yards').sum().alias('receiving_yards_allowed'),
                pl.col('rushing_yards').sum().alias('rushing_yards_allowed'),
                pl.col('receiving_touchdowns').sum().alias('receiving_touchdowns_allowed'),
                pl.col('rushing_touchdowns').sum().alias('rushing_touchdowns_allowed'),
                pl.col('receptions').sum().alias('receptions_allowed'),
                pl.col('targets').sum().alias('targets_allowed'),
                pl.col('rushing_attempts').sum().alias('rushing_attempts_allowed')
            ])
            
            # Calculate league averages and standard deviations
            metrics = ['receiving_yards_allowed', 'rushing_yards_allowed', 
                      'receiving_touchdowns_allowed', 'rushing_touchdowns_allowed',
                      'receptions_allowed', 'targets_allowed', 'rushing_attempts_allowed']
            
            # First calculate per-game stats to account for schedule differences
            defensive_stats = defensive_stats.with_columns([
                pl.col('games_played').count().alias('num_games')
            ])
            
            for metric in metrics:
                # Convert to per-game basis
                defensive_stats = defensive_stats.with_columns([
                    (pl.col(metric) / pl.col('num_games')).alias(f'{metric}_per_game')
                ])
                
                # Calculate league average and std dev
                avg = defensive_stats[f'{metric}_per_game'].mean()
                std = defensive_stats[f'{metric}_per_game'].std()
                
                if std > 0:
                    # Convert to 0-100 scale where:
                    # - 50 = league average
                    # - Each std dev = 15 points
                    # - Better defense = higher score
                    # - Cap at 0-100
                    defensive_stats = defensive_stats.with_columns([
                        pl.lit(50).alias(f'{metric}_score'),  # Start at league average (50)
                        ((avg - pl.col(f'{metric}_per_game')) / std * 15  # Add/subtract points based on std devs
                         + 50).clip(0, 100)  # Clip to 0-100 range
                        .alias(f'{metric}_score')
                    ])
            
            # Calculate composite defensive rating (average of all metric scores)
            score_metrics = [f'{m}_score' for m in metrics]
            defensive_stats = defensive_stats.with_columns([
                pl.fold(0, lambda acc, x: acc + x, [pl.col(m) for m in score_metrics])
                .alias('total_score'),
                pl.lit(len(score_metrics)).alias('num_metrics')
            ]).with_columns([
                (pl.col('total_score') / pl.col('num_metrics')).alias('composite_defense_rating')
            ])
            
            # Convert to dictionary for easy lookup
            return {
                row['opponent']: {
                    'composite_rating': row['composite_defense_rating'],
                    **{metric.replace('_allowed', ''): row[f'{metric}_score'] for metric in metrics}
                }
                for row in defensive_stats.iter_rows(named=True)
            }
            
        except Exception as e:
            logger.error(f"Error calculating defensive rankings for {year}: {str(e)}")
            return {}
    
    def adjust_for_opponent_strength(self, value: float, opponent: str, metric: str, year: int) -> float:
        """
        Adjust a statistical value based on opponent defensive strength.
        
        Args:
            value: The raw statistical value
            opponent: The opposing team
            metric: The metric being adjusted (e.g., 'receiving_yards', 'rushing_touchdowns')
            year: The season year
            
        Returns:
            float: Adjusted value accounting for opponent strength
        
        The adjustment formula:
        adjusted_value = raw_value * (defense_rating / 100)
        
        This means:
        - Against a 100 rated defense (elite): adjusted = raw * 1.0 (100% of raw value)
        - Against a 50 rated defense (average): adjusted = raw * 0.5 (50% of raw value)
        - Against a 25 rated defense (poor): adjusted = raw * 0.25 (25% of raw value)
        
        This reflects that gaining 100 yards against an elite defense (rating=100)
        is equivalent to gaining 200 yards against an average defense (rating=50)
        """
        if year not in self.defense_rankings:
            self.defense_rankings[year] = self.calculate_defensive_rankings(year)
            
        if opponent in self.defense_rankings[year]:
            defense_rating = self.defense_rankings[year][opponent].get(metric, 50)  # Default to average if metric not found
            # Ensure we don't divide by zero or too small numbers
            defense_rating = min(defense_rating, 100)  # Cap maximum rating at 100
            return value * (defense_rating / 100)
        
        return value  # Return unadjusted if no defensive rating available
    
    def calculate_adjusted_metrics(self, df: pl.DataFrame, year: int) -> pl.DataFrame:
        """
        Create opponent-adjusted versions of all offensive metrics.
        Keeps both raw and adjusted versions.
        """
        # List of metrics to adjust
        metrics_to_adjust = [
            'receiving_yards', 'rushing_yards', 
            'receiving_touchdowns', 'rushing_touchdowns',
            'receptions', 'targets', 'rushing_attempts'
        ]
        
        adjusted_cols = []
        for metric in metrics_to_adjust:
            if metric in df.columns:
                # Create adjusted version using opponent defensive rating
                adjusted_cols.append(
                    pl.col(metric).map_elements(
                        lambda x, opponent=pl.col('opponent'): 
                            self.adjust_for_opponent_strength(x, opponent, metric, year)
                    ).alias(f'{metric}_adjusted')
                )
        
        if adjusted_cols:
            return df.with_columns(adjusted_cols)
        return df