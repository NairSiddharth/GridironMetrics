"""
Handles loading and processing of play-by-play data.
Can be extended to use nflverse data in the future.
"""

import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any
import nflreadpy as nfl
from .logger import get_logger
from .constants import CACHE_DIR

logger = get_logger(__name__)

class PlayByPlayProcessor:
    def __init__(self):
        self.pbp_cache: Dict[int, pl.DataFrame] = {}
        
    def load_pbp_data(self, year: int) -> Optional[pl.DataFrame]:
        """
        Load play-by-play data for a given year from nflreadpy.
        Caches the data to avoid repeated loads.
        """
        if year in self.pbp_cache:
            return self.pbp_cache[year]
            
        try:
            logger.info(f"Loading play-by-play data for {year}...")
            pbp_data = nfl.load_pbp(seasons=year)
            
            # Filter to regular season only
            if pbp_data is not None and "season_type" in pbp_data.columns:
                pbp_data = pbp_data.filter(pl.col("season_type") == "REG")
                
            self.pbp_cache[year] = pbp_data
            logger.info(f"Loaded {len(pbp_data)} plays for {year}")
            return pbp_data
        except Exception as e:
            logger.error(f"Error loading PBP data for {year}: {str(e)}")
            return None
    
    def get_situational_context(self, year: int, week: int, team: str) -> Optional[pl.DataFrame]:
        """
        Get situational context for specific team/week with calculated multipliers.
        Returns DataFrame with situational multipliers applied.
        """
        pbp_data = self.load_pbp_data(year)
        if pbp_data is None:
            return None
            
        try:
            # Filter to specific team and week
            team_plays = pbp_data.filter(
                (pl.col("week") == week) & 
                (pl.col("posteam") == team)
            )
            
            if team_plays.height == 0:
                return None
                
            # Calculate situational multipliers
            situational_data = team_plays.with_columns([
                # Field position multipliers
                pl.when(pl.col("yardline_100") <= 5)
                .then(2.0)  # Goalline (highest value)
                .when(pl.col("yardline_100") <= 20)
                .then(1.5)  # Redzone
                .otherwise(1.0)
                .alias("field_position_multiplier"),
                
                # Score differential multipliers
                pl.when(pl.col("score_differential").abs() <= 8)
                .then(1.5)  # One score game
                .when(pl.col("score_differential").abs() <= 16)
                .then(1.25)  # Two score game
                .otherwise(1.0)
                .alias("score_multiplier"),
                
                # Time remaining multipliers (last 2 minutes of each quarter)
                pl.when((pl.col("qtr") == 4) & (pl.col("quarter_seconds_remaining") <= 120))
                .then(1.5)  # 4th quarter critical time (highest)
                .when((pl.col("qtr") == 3) & (pl.col("quarter_seconds_remaining") <= 120))
                .then(1.3)  # 3rd quarter critical time
                .when((pl.col("qtr") == 2) & (pl.col("quarter_seconds_remaining") <= 120))
                .then(1.2)  # 2nd quarter critical time (2-minute drill)
                .when((pl.col("qtr") == 1) & (pl.col("quarter_seconds_remaining") <= 120))
                .then(1.1)  # 1st quarter critical time (lowest)
                .otherwise(1.0)
                .alias("time_multiplier"),
                
                # Down multipliers
                pl.when(pl.col("down") == 4)
                .then(1.5)  # 4th down
                .when(pl.col("down") == 3)
                .then(1.25)  # 3rd down
                .otherwise(1.0)
                .alias("down_multiplier")
            ]).with_columns([
                # Combined situational multiplier (computed after individual multipliers exist)
                (pl.col("field_position_multiplier") * 
                 pl.col("score_multiplier") * 
                 pl.col("time_multiplier") * 
                 pl.col("down_multiplier")).alias("total_situational_multiplier")
            ])
            
            return situational_data
            
        except Exception as e:
            logger.error(f"Error calculating situational context for {team} week {week}: {str(e)}")
            return None