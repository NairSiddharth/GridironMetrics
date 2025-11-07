"""
ML Training Data Builder

Generates training datasets for machine learning models by:
1. Loading historical player stats (2015-2023)
2. Generating features for each player-week using PropFeatureEngineer
3. Matching actual stat outcomes as targets
4. Saving as parquet files for efficient loading

Output:
    cache/ml_training_data/{prop_type}_2015_2023.parquet

Usage:
    builder = TrainingDataBuilder()
    train_df = builder.build_training_dataset(
        start_year=2015,
        end_year=2023,
        prop_type='passing_yards'
    )
"""

import polars as pl
from pathlib import Path
from typing import Optional
from modules.logger import get_logger
from modules.constants import CACHE_DIR
from modules.ml_feature_engineering import PropFeatureEngineer
from modules.prop_types import get_stat_column_for_prop, get_prop_config

logger = get_logger(__name__)


class TrainingDataBuilder:
    """
    Builds training datasets for ML models.

    Generates features for all player-weeks in historical data and matches
    with actual outcomes as targets.
    """

    def __init__(self):
        """Initialize training data builder."""
        self.feature_engineer = PropFeatureEngineer()
        self.output_dir = Path(CACHE_DIR) / "ml_training_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_training_dataset(
        self,
        start_year: int = 2015,
        end_year: int = 2023,
        prop_type: str = 'passing_yards',
        skip_years: list = None
    ) -> pl.DataFrame:
        """
        Generate complete training dataset for a prop type.

        Args:
            start_year: First year to include (default 2015)
            end_year: Last year to include (default 2023)
            prop_type: Prop type (e.g., 'passing_yards')
            skip_years: List of years to exclude (default [2020] - COVID outlier)

        Returns:
            polars DataFrame with feature columns + 'target' column
        """
        # Default to excluding 2020 (COVID outlier year)
        if skip_years is None:
            skip_years = [2020]
        logger.info(f"\n{'='*60}")
        logger.info(f"Building Training Dataset: {prop_type}")
        logger.info(f"Years: {start_year}-{end_year}")
        if skip_years:
            logger.info(f"Excluding years: {skip_years}")
        logger.info(f"{'='*60}")

        # Get prop configuration
        prop_config = get_prop_config(prop_type)
        if not prop_config:
            logger.error(f"Unknown prop type: {prop_type}")
            return pl.DataFrame()

        eligible_positions = prop_config['position']
        stat_column = get_stat_column_for_prop(prop_type)

        logger.info(f"Eligible positions: {eligible_positions}")
        logger.info(f"Target stat column: {stat_column}")

        all_rows = []
        total_examples = 0
        skipped_early_weeks = 0
        skipped_null_targets = 0
        skipped_no_opponent = 0
        skipped_errors = 0

        # Phase 2 Optimization: Pre-load multi-year auxiliary data
        # Load 4 years of data at once to support 3-year historical lookbacks
        logger.info(f"\nPhase 2: Pre-loading multi-year auxiliary data...")
        logger.info(f"  Loading injury, roster, and NextGen data for {start_year-3}-{end_year}")

        multi_year_cache = self._build_multi_year_cache(start_year, end_year)
        logger.info(f"  Multi-year cache built successfully")

        # Process each year
        for year in range(start_year, end_year + 1):
            # Skip excluded years
            if year in skip_years:
                logger.info(f"\nSkipping {year} (excluded year)")
                continue

            logger.info(f"\nProcessing {year}...")
            year_examples = 0

            # Load PBP data once per year (Phase 1 optimization)
            pbp_df = self._load_pbp_data(year)

            # Process each eligible position
            for position in eligible_positions:
                # Load player stats for current year AND 3 prior years (Phase 2 optimization)
                player_stats_cache = {}
                for stats_year in range(year - 3, year + 1):
                    stats_file = Path(CACHE_DIR) / "positional_player_stats" / position.lower() / f"{position.lower()}-{stats_year}.csv"

                    if stats_file.exists():
                        try:
                            stats_df = pl.read_csv(stats_file)
                            player_stats_cache[stats_year] = stats_df
                            if stats_year == year:
                                logger.info(f"  Loaded {len(stats_df)} {position} records for {year}")
                        except Exception as e:
                            logger.debug(f"  Error loading {stats_file}: {e}")

                # Check if current year stats loaded
                if year not in player_stats_cache:
                    logger.warning(f"  Stats file not found for {position} in {year}")
                    continue

                stats_df = player_stats_cache[year]

                # Check if stat column exists
                if stat_column not in stats_df.columns:
                    logger.warning(f"  Stat column '{stat_column}' not found in {position} stats")
                    continue

                # Phase 2: Create data cache with multi-year auxiliary data AND player stats
                # Multi-year cache supports 3-year historical lookbacks without file I/O
                data_cache = {
                    'player_stats': player_stats_cache,            # Multi-year player stats
                    'pbp_df': pbp_df,                              # PBP data for current year
                    'injury_data': multi_year_cache['injury_data'],# Multi-year injury data
                    'roster_data': multi_year_cache['roster_data'],# Multi-year roster data
                    'nextgen_data': multi_year_cache['nextgen_data'] # Multi-year NextGen Stats
                }

                # Process each player-week
                position_examples = 0

                for row in stats_df.iter_rows(named=True):
                    player_id = row.get('player_id')
                    week = row.get('week')
                    team = row.get('team')
                    target_value = row.get(stat_column)

                    # Skip early weeks (need history for features)
                    if week < 4:
                        skipped_early_weeks += 1
                        continue

                    # Skip if target is null
                    if target_value is None:
                        skipped_null_targets += 1
                        continue

                    # Get opponent from PBP data
                    opponent = self._get_opponent_from_pbp(team, year, week, pbp_df)
                    if opponent is None:
                        skipped_no_opponent += 1
                        if position_examples == 0:  # Log first few misses for debugging
                            logger.debug(f"  No opponent found for {team} week {week}")
                        continue

                    # Generate features (Phase 1: pass data_cache to eliminate redundant I/O)
                    try:
                        features = self.feature_engineer.engineer_features(
                            player_id=player_id,
                            season=year,
                            week=week,
                            position=position,
                            prop_type=prop_type,
                            opponent_team=opponent,
                            pbp_df=pbp_df,
                            data_cache=data_cache  # Phase 1 optimization: cached auxiliary data
                        )

                        # Add target
                        features['target'] = float(target_value)

                        # Add metadata for tracking
                        features['player_id'] = player_id
                        features['year'] = year

                        all_rows.append(features)
                        year_examples += 1
                        total_examples += 1
                        position_examples += 1

                    except Exception as e:
                        logger.debug(f"  Error generating features for {player_id} week {week}: {e}")
                        skipped_errors += 1
                        if skipped_errors <= 3:  # Log first few errors
                            logger.warning(f"  Feature generation error sample: {e}")
                        continue

                logger.info(f"  {position}: {position_examples} examples")

            logger.info(f"  Generated {year_examples} examples for {year}")

        # Convert to DataFrame
        if len(all_rows) == 0:
            logger.error("No training examples generated!")
            return pl.DataFrame()

        train_df = pl.DataFrame(all_rows)

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Data Generation Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total examples: {total_examples:,}")
        logger.info(f"Features per example: {len(train_df.columns) - 3}")  # Exclude target, player_id, year
        logger.info(f"\nSkipped:")
        logger.info(f"  Early weeks (< week 4): {skipped_early_weeks:,}")
        logger.info(f"  Null targets: {skipped_null_targets:,}")
        logger.info(f"  No opponent data: {skipped_no_opponent:,}")
        logger.info(f"  Errors: {skipped_errors:,}")

        # Data quality checks
        logger.info(f"\nData Quality:")
        null_counts = train_df.null_count()
        for col in train_df.columns:
            null_pct = (null_counts[col][0] / len(train_df)) * 100
            if null_pct > 0:
                logger.warning(f"  {col}: {null_pct:.1f}% null")

        # Save to parquet
        output_file = self.output_dir / f"{prop_type}_{start_year}_{end_year}.parquet"
        train_df.write_parquet(output_file)
        logger.info(f"\nSaved to: {output_file}")
        logger.info(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

        return train_df

    def _load_pbp_data(self, year: int) -> Optional[pl.DataFrame]:
        """Load play-by-play data for opponent matching."""
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{year}.parquet"

        if not pbp_file.exists():
            logger.warning(f"PBP file not found: {pbp_file}")
            return None

        try:
            pbp = pl.read_parquet(pbp_file)
            return pbp
        except Exception as e:
            logger.error(f"Error loading PBP data: {e}")
            return None

    def _get_opponent_from_pbp(
        self,
        team: str,
        year: int,
        week: int,
        pbp_df: Optional[pl.DataFrame]
    ) -> Optional[str]:
        """
        Get opponent team for a given team-week combination.

        Uses PBP data to find the opponent by looking at games where the team
        was either home or away.
        """
        if pbp_df is None:
            logger.debug(f"No PBP data available for {year}")
            return None

        try:
            # Normalize team code to uppercase
            team_upper = team.upper() if team else None
            if not team_upper:
                return None

            # Check if required columns exist
            if 'home_team' not in pbp_df.columns or 'away_team' not in pbp_df.columns:
                logger.debug(f"Missing home_team/away_team columns in PBP data")
                return None

            # Find games for this team-week
            team_games = pbp_df.filter(
                (pl.col('week') == week) &
                ((pl.col('home_team') == team_upper) | (pl.col('away_team') == team_upper))
            ).select(['home_team', 'away_team']).unique()

            if len(team_games) == 0:
                return None

            # Get first game (should only be one)
            game = team_games.row(0, named=True)
            home_team = game['home_team']
            away_team = game['away_team']

            # Return opponent
            if home_team == team_upper:
                return away_team
            else:
                return home_team

        except Exception as e:
            logger.debug(f"Error getting opponent for {team} week {week}: {e}")
            return None

    def _build_multi_year_cache(self, start_year: int, end_year: int) -> dict:
        """
        Build multi-year cache of auxiliary data for historical lookbacks.

        Phase 2 optimization: Pre-load 4 years of auxiliary data at once
        to support 3-year historical lookbacks without file I/O.

        Args:
            start_year: First year of training data
            end_year: Last year of training data

        Returns:
            Dictionary with multi-year cached data:
            {
                'injury_data': {2023: df, 2022: df, ...},
                'roster_data': {2023: df, 2022: df, ...},
                'nextgen_data': {2023: df, 2022: df, ...}
            }
        """
        # Load 4 years of data (current + 3 prior) to support 3-year lookbacks
        years_to_load = range(start_year - 3, end_year + 1)

        injury_cache = {}
        roster_cache = {}
        nextgen_cache = {}

        for year in years_to_load:
            # Load injury data
            injury_data = self._load_injury_data_cached(year)
            if injury_data is not None:
                injury_cache[year] = injury_data

            # Load roster data
            roster_data = self._load_roster_data_cached(year)
            if roster_data is not None:
                roster_cache[year] = roster_data

            # Load NextGen data
            nextgen_data = self._load_nextgen_data_cached(year)
            if nextgen_data is not None:
                nextgen_cache[year] = nextgen_data

        logger.info(f"  Loaded {len(injury_cache)} years of injury data")
        logger.info(f"  Loaded {len(roster_cache)} years of roster data")
        logger.info(f"  Loaded {len(nextgen_cache)} years of NextGen data")

        return {
            'injury_data': injury_cache,
            'roster_data': roster_cache,
            'nextgen_data': nextgen_cache
        }

    def _load_injury_data_cached(self, year: int) -> Optional[pl.DataFrame]:
        """
        Load injury data once per year for caching.

        Phase 1 optimization: Load auxiliary data files once per year
        instead of loading them thousands of times per player-week.

        Args:
            year: Season year

        Returns:
            DataFrame with injury data or None if not available
        """
        try:
            from modules.injury_cache_builder import load_injury_data
            return load_injury_data(year)
        except Exception as e:
            logger.debug(f"Error loading injury data for {year}: {e}")
            return None

    def _load_roster_data_cached(self, year: int) -> Optional[pl.DataFrame]:
        """
        Load roster data once per year for caching.

        Phase 1 optimization: Load auxiliary data files once per year
        instead of loading them thousands of times per player-week.

        Args:
            year: Season year

        Returns:
            DataFrame with roster data or None if not available
        """
        try:
            from modules.roster_cache_builder import load_roster_data
            return load_roster_data(year)
        except Exception as e:
            logger.debug(f"Error loading roster data for {year}: {e}")
            return None

    def _load_nextgen_data_cached(self, year: int) -> Optional[pl.DataFrame]:
        """
        Load NextGen Stats data once per year for caching.

        Phase 1 optimization: Load auxiliary data files once per year
        instead of loading them thousands of times per player-week.

        Args:
            year: Season year

        Returns:
            DataFrame with NextGen Stats data or None if not available
        """
        try:
            from modules.nextgen_cache_builder import load_nextgen_cache
            return load_nextgen_cache(year)
        except Exception as e:
            logger.debug(f"Error loading NextGen data for {year}: {e}")
            return None

    def build_all_prop_types(
        self,
        start_year: int = 2015,
        end_year: int = 2023,
        skip_years: list = None
    ):
        """
        Build training datasets for all 7 prop types.

        Args:
            start_year: First year to include
            end_year: Last year to include
            skip_years: List of years to exclude (default [2020] - COVID outlier)
        """
        prop_types = [
            'passing_yards',
            'passing_tds',
            'rushing_yards',
            'rushing_tds',
            'receptions',
            'receiving_yards',
            'receiving_tds'
        ]

        results = {}

        for prop_type in prop_types:
            logger.info(f"\n\n{'#'*60}")
            logger.info(f"# Building: {prop_type.upper()}")
            logger.info(f"{'#'*60}\n")

            try:
                train_df = self.build_training_dataset(start_year, end_year, prop_type, skip_years)
                results[prop_type] = {
                    'examples': len(train_df),
                    'features': len(train_df.columns) - 3,  # Exclude target, player_id, year
                    'status': 'SUCCESS'
                }
            except Exception as e:
                logger.error(f"Error building {prop_type}: {e}")
                results[prop_type] = {
                    'examples': 0,
                    'features': 0,
                    'status': f'FAILED: {e}'
                }

        # Print summary
        logger.info(f"\n\n{'='*60}")
        logger.info("ALL PROP TYPES SUMMARY")
        logger.info(f"{'='*60}")

        total_examples = 0
        for prop_type, result in results.items():
            logger.info(f"{prop_type:20s}: {result['examples']:,} examples ({result['status']})")
            total_examples += result['examples']

        logger.info(f"\nTotal examples across all props: {total_examples:,}")
        logger.info(f"{'='*60}")

        return results


if __name__ == "__main__":
    # Test on 2023 data only (faster for initial validation)
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*60)
    print("ML Training Data Builder - Test")
    print("="*60)
    print("\nTesting on 2023 data only (for speed)")
    print("="*60 + "\n")

    builder = TrainingDataBuilder()

    # Build passing_yards dataset for 2023 only
    train_df = builder.build_training_dataset(
        start_year=2023,
        end_year=2023,
        prop_type='passing_yards'
    )

    if len(train_df) > 0:
        print("\n" + "="*60)
        print("SAMPLE DATA (first 5 rows)")
        print("="*60)
        print(f"Columns: {train_df.columns}")
        print(f"Shape: {train_df.shape}")
        # Skip printing full dataframe to avoid encoding issues

        print("\n" + "="*60)
        print("TARGET STATISTICS")
        print("="*60)
        print(f"Mean: {train_df['target'].mean():.2f}")
        print(f"Std: {train_df['target'].std():.2f}")
        print(f"Min: {train_df['target'].min():.2f}")
        print(f"Max: {train_df['target'].max():.2f}")

    print("\n" + "="*60)
    print("Training data builder ready")
    print("="*60)
