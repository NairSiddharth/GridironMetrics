"""
Handles loading and processing of play-by-play data.
Loads from pre-built cache for performance. Use pbp_cache_builder.py to build cache.
"""

import polars as pl
from pathlib import Path
from typing import Optional, Dict
from .logger import get_logger
from .constants import CACHE_DIR

logger = get_logger(__name__)

class PlayByPlayProcessor:
    def __init__(self):
        """Initialize the PBP processor."""
        self.pbp_cache: Dict[int, pl.DataFrame] = {}
        self.cache_dir = Path(CACHE_DIR) / "pbp"
        
    def load_pbp_data(self, year: int) -> Optional[pl.DataFrame]:
        """
        Load play-by-play data for a given year.
        First tries to load from cache, falls back to nflreadpy if cache not found.
        """
        if year in self.pbp_cache:
            return self.pbp_cache[year]
        
        # Try loading from cache first
        cache_path = self.cache_dir / f"pbp_{year}.parquet"
        if cache_path.exists():
            try:
                logger.info(f"Loading play-by-play data for {year} from cache...")
                pbp_data = pl.read_parquet(cache_path)

                # Normalize team codes to match player/team stats (lowercase)
                team_columns = ['posteam', 'defteam', 'home_team', 'away_team']
                for col in team_columns:
                    if col in pbp_data.columns:
                        pbp_data = pbp_data.with_columns([
                            pl.col(col).str.to_lowercase().alias(col)
                        ])

                self.pbp_cache[year] = pbp_data
                logger.info(f"Loaded {len(pbp_data)} plays for {year} from cache")
                return pbp_data
            except Exception as e:
                logger.error(f"Error loading from cache for {year}: {str(e)}")
                logger.info("Falling back to nflreadpy...")
        else:
            logger.warning(f"Cache not found for {year}: {cache_path}")
            logger.warning("Run 'python -m modules.pbp_cache_builder' to build cache for better performance")
            logger.info("Falling back to nflreadpy (this will be slow)...")
        
        # Fall back to nflreadpy if cache not available
        try:
            import nflreadpy as nfl
            logger.info(f"Loading play-by-play data for {year} from nflreadpy...")
            pbp_data = nfl.load_pbp(seasons=year)
            
            # Filter to regular season only
            if pbp_data is not None and "season_type" in pbp_data.columns:
                pbp_data = pbp_data.filter(pl.col("season_type") == "REG")

            # Normalize team codes to match player/team stats (lowercase)
            team_columns = ['posteam', 'defteam', 'home_team', 'away_team']
            for col in team_columns:
                if col in pbp_data.columns:
                    pbp_data = pbp_data.with_columns([
                        pl.col(col).str.to_lowercase().alias(col)
                    ])

            # Calculate multipliers since they won't be in non-cached data
            pbp_data = self._calculate_multipliers(pbp_data)
                
            self.pbp_cache[year] = pbp_data
            logger.info(f"Loaded {len(pbp_data)} plays for {year}")
            return pbp_data
        except Exception as e:
            logger.error(f"Error loading PBP data for {year}: {str(e)}")
            return None
    
    def _calculate_multipliers(self, pbp_data: pl.DataFrame) -> pl.DataFrame:
        """Calculate situational multipliers for PBP data (used when loading without cache)."""
        return pbp_data.with_columns([
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
            # Combined situational multiplier
            (pl.col("field_position_multiplier") * 
             pl.col("score_multiplier") * 
             pl.col("time_multiplier") * 
             pl.col("down_multiplier")).alias("situational_multiplier")
        ])
    
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
            
            # Multipliers should already be calculated (either from cache or _calculate_multipliers)
            return team_plays
            
        except Exception as e:
            logger.error(f"Error getting situational context for {team} week {week}: {str(e)}")
            return None