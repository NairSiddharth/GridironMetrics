"""
Data Loaders Module

Functions for loading cached NFL data from disk.
"""

from pathlib import Path
from datetime import datetime
import polars as pl
from modules.logger import get_logger
from modules.constants import CACHE_DIR, normalize_team_codes_in_dataframe, TEAM_RELOCATIONS

logger = get_logger(__name__)

def is_season_complete(year: int) -> bool:
    """
    Check if a season is complete based on current date.

    Args:
        year: Season year to check

    Returns:
        True if season is complete, False if ongoing
    """
    current_year = datetime.now().year
    current_month = datetime.now().month

    # Previous years are always complete
    if year < current_year:
        return True

    # Current year: complete if February or later (after Super Bowl)
    if year == current_year and current_month >= 2:
        return True

    return False

def classify_player_profile(boom_count: int, steady_count: int, bust_count: int) -> str:
    """
    Classify player's overall performance profile based on game distribution.

    Args:
        boom_count: Number of boom games (>0.5 SD above mean)
        steady_count: Number of steady games (within ±0.5 SD of mean)
        bust_count: Number of bust games (<0.5 SD below mean)

    Returns:
        Profile classification string
    """
    total_games = boom_count + steady_count + bust_count

    if total_games == 0:
        return "N/A"

    steady_pct = steady_count / total_games
    volatile_pct = (boom_count + bust_count) / total_games

    # Classify based on distribution patterns
    if steady_pct >= 0.50:
        return "Consistent"
    elif volatile_pct >= 0.75:
        return "Boom/Bust"
    elif boom_count > bust_count * 1.5:
        return "High-Ceiling"
    elif bust_count > boom_count * 1.5:
        return "Low-Floor"
    else:
        return "Volatile"

def load_team_weekly_stats(year: int) -> pl.DataFrame:
    """Load team weekly stats for a given year, handling team relocations."""
    try:
        # Get all team data from the cache
        team_data = []
        team_cache_dir = Path(CACHE_DIR) / "team_stats"
        loaded_teams = set()  # Track which teams we've loaded to avoid duplicates

        # For each team directory, try to load data for this year
        for team_dir in team_cache_dir.iterdir():
            if team_dir.is_dir():
                team_code = team_dir.name

                # Skip if we've already loaded this team (handles relocation cases)
                if team_code in loaded_teams:
                    continue

                file_path = team_dir / f"{team_code}-{year}.csv"
                if file_path.exists():
                    try:
                        df = pl.read_csv(file_path, infer_schema_length=10000)

                        # Normalize team codes in the dataframe
                        df = normalize_team_codes_in_dataframe(df, year)

                        team_data.append(df)
                        loaded_teams.add(team_code)

                        # For relocated teams, mark the other code as loaded too
                        for old_code, info in TEAM_RELOCATIONS.items():
                            if team_code == old_code:
                                loaded_teams.add(info["new_code"])
                            elif team_code == info["new_code"]:
                                loaded_teams.add(old_code)

                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {str(e)}")
                        continue
        
        if not team_data:
            logger.error(f"No team data found for {year}")
            return None
        
        # Cast all dataframes to have consistent schema
        try:
            # Get union of all columns
            all_cols = set()
            for df in team_data:
                all_cols.update(df.columns)
            
            # For each dataframe, add missing columns as nulls and cast to strings where needed
            aligned_data = []
            for df in team_data:
                # Add missing columns as null
                for col in all_cols:
                    if col not in df.columns:
                        df = df.with_columns(pl.lit(None).alias(col))
                
                # Cast all columns to string temporarily for concat
                schema = {col: pl.Utf8 for col in df.columns}
                df = df.cast(schema, strict=False)
                aligned_data.append(df)
            
            # Concat all dataframes
            combined_df = pl.concat(aligned_data, how="vertical")
            
            # Cast numeric columns back
            numeric_cols = ['week', 'completions', 'attempts', 'passing_yards', 'passing_tds', 
                          'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds',
                          'receptions', 'targets', 'carries', 'sacks', 'interceptions',
                          'def_tds', 'special_teams_tds', 'fg_made', 'pat_made']
            for col in numeric_cols:
                if col in combined_df.columns:
                    combined_df = combined_df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
            
            return combined_df
        except Exception as e:
            logger.error(f"Error concatenating team data for {year}: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error loading team stats for {year}: {str(e)}")
        return None

def load_position_weekly_stats(year: int, position: str) -> pl.DataFrame:
    """Load weekly stats for a specific position and year."""
    try:
        file_path = Path(CACHE_DIR) / "positional_player_stats" / position.lower() / f"{position.lower()}-{year}.csv"
        if not file_path.exists():
            logger.error(f"No {position} data found for {year}")
            return None

        df = pl.read_csv(file_path)

        # Convert problematic list columns to String type
        for col in ['fg_blocked_list', 'fg_missed_list', 'fg_made_list',
                   'fg_made_distance', 'fg_missed_distance', 'fg_blocked_distance', 'gwfg_distance']:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False))

        # Convert columns that should be Float64 but might be String
        float_columns = ['pacr', 'passing_cpoe', 'passing_epa', 'racr', 'rushing_epa', 'receiving_epa']
        for col in float_columns:
            if col in df.columns and df[col].dtype == pl.Utf8:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        # Normalize team codes to handle relocations
        df = normalize_team_codes_in_dataframe(df, year)

        return df
    except Exception as e:
        logger.error(f"Error loading {position} stats for {year}: {str(e)}")
        return None
