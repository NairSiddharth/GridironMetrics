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

            # OPTIMIZATION: Pre-compute injury metrics for all players (Phase 3)
            # Eliminates ~10,000 N+1 queries per year (15-20% speedup)
            injury_metrics_batch = self._precompute_injury_metrics_batch(
                year=year,
                multi_year_cache=multi_year_cache,
                eligible_positions=eligible_positions
            )
            logger.info(f"  Pre-computed injury metrics for {len(injury_metrics_batch)} player-years")

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

                # Phase 2-8: Create data cache with multi-year auxiliary data AND player stats
                # Multi-year cache supports 3-year historical lookbacks without file I/O
                # Phase 3 adds pre-computed injury metrics to eliminate N+1 queries
                # Phase 4 adds game metadata cache to eliminate 30k+ PBP filter operations
                # Phase 5 adds player volume cache to eliminate 10k+ PBP filter operations
                # Phase 6 adds data_cache to prior season features (file I/O fix)
                # Phase 7 adds PBP-derived stats cache (catch rate, efficiency)
                # Phase 8 adds game script & opponent defense cache (eliminates 10+ filters per player-week)
                data_cache = {
                    'player_stats': player_stats_cache,            # Multi-year player stats
                    'pbp_df': pbp_df,                              # PBP data for current year
                    'injury_data': multi_year_cache['injury_data'],# Multi-year injury data
                    'roster_data': multi_year_cache['roster_data'],# Multi-year roster data
                    'nextgen_data': multi_year_cache['nextgen_data'], # Multi-year NextGen Stats
                    'betting_lines': multi_year_cache.get('betting_lines', {}), # Betting lines data
                    'injury_metrics_batch': injury_metrics_batch,  # Pre-computed injury metrics (Phase 3)
                    'game_metadata': multi_year_cache.get('game_metadata', {}), # Game-level metadata (Phase 4)
                    'player_volume': multi_year_cache.get('player_volume', {}),  # Player volume stats (Phase 5)
                    'pbp_stats': multi_year_cache.get('pbp_stats', {}),  # PBP-derived stats (Phase 7)
                    'game_script': multi_year_cache.get('game_script', {}),  # Game script stats (Phase 8A)
                    'opponent_defense': multi_year_cache.get('opponent_defense', {})  # Opponent defense stats (Phase 8B)
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
                        features['week'] = week

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
        betting_cache = {}
        snap_count_cache = {}

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

            # Load betting lines
            betting_data = self._load_betting_lines_cached(year)
            if betting_data is not None:
                betting_cache[year] = betting_data

            # Load snap count data
            snap_count_data = self._load_snap_count_data_cached(year)
            if snap_count_data is not None:
                snap_count_cache[year] = snap_count_data

        # Phase 4: Load game metadata for fast lookups
        game_metadata_cache = {}
        for year in range(start_year, end_year + 1):
            pbp_df = self._load_pbp_data(year)
            game_metadata = self._build_game_metadata_cache(year, pbp_df)
            if game_metadata:
                game_metadata_cache[year] = game_metadata

        # Merge betting lines into game metadata
        for year, betting_data in betting_cache.items():
            if year in game_metadata_cache:
                for (team, week), betting_info in betting_data.items():
                    if (team, week) in game_metadata_cache[year]:
                        game_metadata_cache[year][(team, week)]['vegas_total'] = betting_info['vegas_total']
                        game_metadata_cache[year][(team, week)]['vegas_spread'] = betting_info['vegas_spread']

        # Phase 5: Load player volume stats for fast lookups
        player_volume_cache = {}
        for year in range(start_year, end_year + 1):
            pbp_df = self._load_pbp_data(year)
            player_volume = self._build_player_volume_cache(year, pbp_df)
            if player_volume:
                player_volume_cache[year] = player_volume

        # Phase 7: Load PBP-derived stats (catch rate, efficiency) for fast lookups
        pbp_stats_cache = {}
        for year in range(start_year, end_year + 1):
            pbp_df = self._load_pbp_data(year)
            pbp_stats = self._build_pbp_stats_cache(year, pbp_df)
            if pbp_stats:
                pbp_stats_cache[year] = pbp_stats

        # Phase 8A: Load game script stats (team margin, pace, TOP, opp defense) for fast lookups
        game_script_cache = {}
        for year in range(start_year, end_year + 1):
            pbp_df = self._load_pbp_data(year)
            game_script = self._build_game_script_cache(year, pbp_df)
            if game_script:
                game_script_cache[year] = game_script

        # Phase 8B: Load opponent defense stats for fast lookups
        opponent_defense_cache = {}
        for year in range(start_year, end_year + 1):
            pbp_df = self._load_pbp_data(year)
            opponent_defense = self._build_opponent_defense_cache(year, pbp_df)
            if opponent_defense:
                opponent_defense_cache[year] = opponent_defense

        # Phase 9: Load player ID mapping (one-time, not per year)
        player_id_mapping = self._load_player_id_mapping_cached()

        logger.info(f"  Loaded {len(injury_cache)} years of injury data")
        logger.info(f"  Loaded {len(roster_cache)} years of roster data")
        logger.info(f"  Loaded {len(nextgen_cache)} years of NextGen data")
        logger.info(f"  Loaded {len(betting_cache)} years of betting lines")
        logger.info(f"  Loaded {len(snap_count_cache)} years of snap count data")
        logger.info(f"  Loaded {len(game_metadata_cache)} years of game metadata")
        logger.info(f"  Loaded {len(player_volume_cache)} years of player volume stats")
        logger.info(f"  Loaded {len(pbp_stats_cache)} years of PBP-derived stats")
        logger.info(f"  Loaded {len(game_script_cache)} years of game script stats")
        logger.info(f"  Loaded {len(opponent_defense_cache)} years of opponent defense stats")
        if player_id_mapping is not None:
            logger.info(f"  Loaded player ID mapping ({len(player_id_mapping)} players)")

        return {
            'injury_data': injury_cache,
            'roster_data': roster_cache,
            'nextgen_data': nextgen_cache,
            'betting_lines': betting_cache,
            'snap_count_data': snap_count_cache,
            'player_id_mapping': player_id_mapping,
            'game_metadata': game_metadata_cache,
            'player_volume': player_volume_cache,
            'pbp_stats': pbp_stats_cache,
            'game_script': game_script_cache,
            'opponent_defense': opponent_defense_cache
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

    def _load_snap_count_data_cached(self, year: int) -> Optional[pl.DataFrame]:
        """
        Load snap count data once per year for caching.

        Phase 9 optimization: Load snap count data files once per year
        to support route participation features.

        Args:
            year: Season year

        Returns:
            DataFrame with snap count data or None if not available
        """
        try:
            from modules.snap_count_cache_builder import load_snap_count_cache
            return load_snap_count_cache(year)
        except Exception as e:
            logger.debug(f"Error loading snap count data for {year}: {e}")
            return None

    def _load_player_id_mapping_cached(self) -> Optional[pl.DataFrame]:
        """
        Load player ID mapping (one-time load).

        Mapping between pfr_player_id (snap counts) and gsis_id (main system).

        Returns:
            DataFrame with player ID mapping or None if not available
        """
        try:
            mapping_file = Path(CACHE_DIR) / "player_id_mapping.parquet"
            if not mapping_file.exists():
                logger.debug("Player ID mapping not found")
                return None
            return pl.read_parquet(mapping_file)
        except Exception as e:
            logger.debug(f"Error loading player ID mapping: {e}")
            return None

    def _load_betting_lines_cached(self, year: int) -> Optional[dict]:
        """
        Load betting lines once per year for caching.

        Creates a lookup dict: (team_abbr, week) -> {vegas_total, vegas_spread}

        Args:
            year: Season year

        Returns:
            Dict with betting data or None if not available
        """
        try:
            import json

            betting_file = Path(CACHE_DIR) / "betting_lines" / f"{year}.json"
            if not betting_file.exists():
                return None

            with open(betting_file) as f:
                games = json.load(f)

            # Team name mapping: betting lines -> PBP abbreviations
            TEAM_MAP = {
                "Chiefs": "KC", "Ravens": "BAL", "Bills": "BUF", "Bengals": "CIN",
                "Browns": "CLE", "Broncos": "DEN", "Texans": "HOU", "Colts": "IND",
                "Jaguars": "JAX", "Raiders": "LV", "Chargers": "LAC", "Dolphins": "MIA",
                "Patriots": "NE", "Jets": "NYJ", "Steelers": "PIT", "Titans": "TEN",
                "Cowboys": "DAL", "Giants": "NYG", "Eagles": "PHI", "Washington": "WAS",
                "Bears": "CHI", "Lions": "DET", "Packers": "GB", "Vikings": "MIN",
                "Falcons": "ATL", "Panthers": "CAR", "Saints": "NO", "Buccaneers": "TB",
                "Cardinals": "ARI", "Rams": "LA", "49ers": "SF", "Seahawks": "SEA"
            }

            # Load PBP to get date -> week mapping
            pbp_df = self._load_pbp_data(year)
            if pbp_df is None:
                return None

            # Create date -> (week, home_team, away_team) mapping
            date_week_map = {}
            games_pbp = pbp_df.select(['game_date', 'week', 'home_team', 'away_team']).unique(
                subset=['game_date', 'home_team', 'away_team']
            )

            for row in games_pbp.iter_rows(named=True):
                game_date = row['game_date'].replace('-', '')  # "2024-09-06" -> "20240906"
                home = row['home_team']
                away = row['away_team']
                week = row['week']
                date_week_map[(game_date, home, away)] = week

            # Build lookup: (team_abbr, week) -> {vegas_total, vegas_spread}
            betting_lookup = {}
            for game in games:
                date_str = str(int(game['date']))  # Convert float to int to remove ".0"
                home_name = game['home_team']
                away_name = game['away_team']

                # Convert betting names to PBP abbreviations
                home = TEAM_MAP.get(home_name)
                away = TEAM_MAP.get(away_name)

                if not home or not away:
                    continue

                # Find week for this game
                week = date_week_map.get((date_str, home, away))
                if week is None:
                    continue

                # Extract betting lines
                vegas_total = game.get('close_over_under')
                home_spread = game.get('home_close_spread')
                away_spread = game.get('away_close_spread')

                if vegas_total is None or home_spread is None or away_spread is None:
                    continue

                # Store for home team perspective
                betting_lookup[(home, week)] = {
                    'vegas_total': float(vegas_total),
                    'vegas_spread': float(home_spread)  # negative = favorite
                }

                # Store for away team perspective
                betting_lookup[(away, week)] = {
                    'vegas_total': float(vegas_total),
                    'vegas_spread': float(away_spread)  # negative = favorite
                }

            return betting_lookup

        except Exception as e:
            logger.debug(f"Error loading betting lines for {year}: {e}")
            return None

    def _build_game_metadata_cache(self, year: int, pbp_df: Optional[pl.DataFrame]) -> dict:
        """
        Extract game metadata once per game instead of filtering per player-week.

        Phase 4 optimization: Pre-compute game-level metadata (home/away, weather, venue)
        to eliminate 30,000+ redundant PBP filter operations.

        Args:
            year: Season year
            pbp_df: Pre-loaded PBP DataFrame for the year

        Returns:
            Dict mapping (team, week) -> {home_team, away_team, is_home, is_dome, temp, wind, opponent}

        Example:
            cache[('LAC', 1)] = {'home_team': 'LAC', 'away_team': 'MIA', 'is_home': 1.0, ...}
            cache[('MIA', 1)] = {'home_team': 'LAC', 'away_team': 'MIA', 'is_home': 0.0, ...}
        """
        if pbp_df is None or len(pbp_df) == 0:
            return {}

        try:
            # Extract unique games (one row per game)
            # Select only the columns we need to avoid memory issues
            games = pbp_df.select([
                'game_id', 'week', 'home_team', 'away_team',
                'roof', 'temp', 'wind', 'div_game'
            ]).unique(subset=['game_id'])

            game_cache = {}

            for game in games.iter_rows(named=True):
                home_team = game['home_team']
                away_team = game['away_team']
                week = game['week']

                if not home_team or not away_team:
                    continue

                # Common metadata
                roof = game['roof']
                temp = game['temp']
                wind = game['wind']
                div_game = game['div_game']

                # Null out weather for indoor games (dome/closed) - outdoor ambient temps are misleading
                is_indoor = roof in ['dome', 'closed']

                metadata = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'roof': roof,
                    'temp': None if is_indoor else (float(temp) if temp is not None else None),
                    'wind': None if is_indoor else (float(wind) if wind is not None else None),
                    'is_dome': 1.0 if is_indoor else 0.0 if roof else None,
                    'div_game': float(div_game) if div_game is not None else None
                }

                # Home team entry
                game_cache[(home_team, week)] = {
                    **metadata,
                    'is_home': 1.0,
                    'opponent': away_team
                }

                # Away team entry
                game_cache[(away_team, week)] = {
                    **metadata,
                    'is_home': 0.0,
                    'opponent': home_team
                }

            logger.debug(f"  Built game metadata cache for {year}: {len(game_cache)} team-week entries ({len(games)} games)")
            return game_cache

        except Exception as e:
            logger.warning(f"Error building game metadata cache for {year}: {e}")
            return {}

    def _build_player_volume_cache(self, year: int, pbp_df: Optional[pl.DataFrame]) -> dict:
        """
        Pre-compute player volume stats once per season to eliminate repeated PBP filtering.

        Phase 5 optimization: Build player-week volume cache for O(1) lookups.
        Eliminates 10,000+ redundant PBP filter operations per year.

        Args:
            year: Season year
            pbp_df: Pre-loaded PBP DataFrame for the year

        Returns:
            Dict mapping (player_id, week) -> {
                'targets': int,
                'receptions': int,
                'receiving_yards': float,
                'carries': int,
                'rushing_yards': float,
                'team': str,
                'position': str,
                'team_targets': int,  # Total team targets through this week
                'team_carries': int,  # Total team carries through this week
            }
        """
        if pbp_df is None or len(pbp_df) == 0:
            return {}

        try:
            volume_cache = {}

            # Process all weeks
            max_week = pbp_df['week'].max()

            for current_week in range(1, max_week + 1):
                # Filter to data through current week
                pbp_through_week = pbp_df.filter(pl.col('week') <= current_week)

                # === PASSING TARGETS (WR/TE/RB) ===
                # Get all targets (complete + incomplete passes)
                targets = pbp_through_week.filter(
                    ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1)) &
                    pl.col('receiver_player_id').is_not_null()
                )

                if len(targets) > 0:
                    # Group by player to get cumulative stats
                    player_targets = targets.group_by('receiver_player_id').agg([
                        pl.count().alias('targets'),
                        pl.col('complete_pass').sum().alias('receptions'),
                        pl.col('receiving_yards').fill_null(0).sum().alias('receiving_yards'),
                        pl.col('posteam').first().alias('team')
                    ])

                    # Get team totals for target share
                    team_targets = targets.group_by('posteam').agg([
                        pl.count().alias('team_targets')
                    ])

                    # Merge team totals
                    player_targets = player_targets.join(
                        team_targets,
                        left_on='team',
                        right_on='posteam',
                        how='left'
                    )

                    # Store in cache
                    for row in player_targets.iter_rows(named=True):
                        player_id = row['receiver_player_id']
                        cache_key = (player_id, current_week)

                        volume_cache[cache_key] = {
                            'targets': row['targets'],
                            'receptions': row['receptions'],
                            'receiving_yards': row['receiving_yards'],
                            'team': row['team'],
                            'team_targets': row['team_targets'],
                            'carries': 0,  # Will be updated below for RBs
                            'rushing_yards': 0.0,
                            'team_carries': 0
                        }

                # === RUSHING CARRIES (RB/QB) ===
                carries = pbp_through_week.filter(
                    (pl.col('rush_attempt') == 1) &
                    pl.col('rusher_player_id').is_not_null()
                )

                if len(carries) > 0:
                    # Group by player
                    player_carries = carries.group_by('rusher_player_id').agg([
                        pl.count().alias('carries'),
                        pl.col('rushing_yards').fill_null(0).sum().alias('rushing_yards'),
                        pl.col('posteam').first().alias('team')
                    ])

                    # Get team totals for carry share
                    team_carries = carries.group_by('posteam').agg([
                        pl.count().alias('team_carries')
                    ])

                    # Merge team totals
                    player_carries = player_carries.join(
                        team_carries,
                        left_on='team',
                        right_on='posteam',
                        how='left'
                    )

                    # Store in cache (update existing or create new)
                    for row in player_carries.iter_rows(named=True):
                        player_id = row['rusher_player_id']
                        cache_key = (player_id, current_week)

                        if cache_key in volume_cache:
                            # Update existing entry (player has both targets and carries)
                            volume_cache[cache_key]['carries'] = row['carries']
                            volume_cache[cache_key]['rushing_yards'] = row['rushing_yards']
                            volume_cache[cache_key]['team_carries'] = row['team_carries']
                        else:
                            # Create new entry (RB with no targets)
                            volume_cache[cache_key] = {
                                'targets': 0,
                                'receptions': 0,
                                'receiving_yards': 0.0,
                                'team': row['team'],
                                'team_targets': 0,
                                'carries': row['carries'],
                                'rushing_yards': row['rushing_yards'],
                                'team_carries': row['team_carries']
                            }

            logger.debug(f"  Built player volume cache for {year}: {len(volume_cache)} player-week entries")
            return volume_cache

        except Exception as e:
            logger.warning(f"Error building player volume cache for {year}: {e}")
            return {}

    def _build_pbp_stats_cache(self, year: int, pbp_df: Optional[pl.DataFrame]) -> dict:
        """
        Pre-compute CUMULATIVE PBP-derived stats (catch rate, efficiency) per player-week.

        Phase 7 optimization: Stores cumulative stats (not per-week) for O(1) lookups.
        Eliminates 10,000+ redundant PBP filter operations AND aggregation loops.

        Args:
            year: Season year
            pbp_df: Pre-loaded PBP DataFrame for the year

        Returns:
            Dict mapping (player_id, week) -> {
                'targets': int (cumulative through this week),
                'completions': int,
                'air_yards_total': float,
                'air_yards_count': int,
                'yac_total': float,
                'receiving_yards_total': float,
                'completions': int (successful plays),
                'red_zone_targets': int,
            }
        """
        if pbp_df is None or len(pbp_df) == 0:
            return {}

        try:
            pbp_cache = {}
            max_week = pbp_df['week'].max()

            for current_week in range(1, max_week + 1):
                # Filter to data THROUGH current week (cumulative)
                pbp_through_week = pbp_df.filter(pl.col('week') <= current_week)

                # === RECEIVING STATS (WR/TE/RB) ===
                if 'receiver_player_id' in pbp_through_week.columns:
                    # All targets through this week
                    targets = pbp_through_week.filter(
                        ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1)) &
                        pl.col('receiver_player_id').is_not_null()
                    )

                    if len(targets) > 0:
                        # Group by player for cumulative stats
                        player_stats = targets.group_by('receiver_player_id').agg([
                            pl.count().alias('targets'),
                            pl.col('complete_pass').sum().alias('completions'),
                            pl.col('air_yards').sum().alias('air_yards_total'),
                            pl.col('air_yards').count().alias('air_yards_count'),
                            pl.col('yards_after_catch').sum().alias('yac_total'),
                            pl.col('receiving_yards').sum().alias('receiving_yards_total'),
                        ])

                        # Add red zone targets (for efficiency features)
                        if 'yardline_100' in pbp_through_week.columns:
                            red_zone_targets = targets.filter(pl.col('yardline_100') <= 20).group_by('receiver_player_id').agg([
                                pl.count().alias('red_zone_targets')
                            ])
                            player_stats = player_stats.join(red_zone_targets, on='receiver_player_id', how='left')

                        # Add success rate (if available)
                        if 'success' in pbp_through_week.columns:
                            completions = targets.filter(pl.col('complete_pass') == 1)
                            if len(completions) > 0:
                                success_stats = completions.group_by('receiver_player_id').agg([
                                    pl.col('success').sum().alias('successful_plays'),
                                    pl.count().alias('total_completions_check')
                                ])
                                player_stats = player_stats.join(success_stats, on='receiver_player_id', how='left')

                        for row in player_stats.iter_rows(named=True):
                            player_id = row['receiver_player_id']
                            cache_key = (player_id, current_week)

                            pbp_cache[cache_key] = {
                                'targets': row['targets'],
                                'completions': row['completions'],
                                'air_yards_total': row['air_yards_total'] if row['air_yards_total'] else 0.0,
                                'air_yards_count': row['air_yards_count'],
                                'yac_total': row['yac_total'] if row['yac_total'] else 0.0,
                                'receiving_yards_total': row['receiving_yards_total'] if row['receiving_yards_total'] else 0.0,
                                'red_zone_targets': row.get('red_zone_targets', 0) or 0,
                                'successful_plays': row.get('successful_plays', 0) or 0,
                            }

            logger.debug(f"  Built PBP stats cache for {year}: {len(pbp_cache)} player-week entries (cumulative)")
            return pbp_cache

        except Exception as e:
            logger.warning(f"Error building PBP stats cache for {year}: {e}")
            return {}

    def _build_game_script_cache(self, year: int, pbp_df: Optional[pl.DataFrame]) -> dict:
        """
        Pre-compute CUMULATIVE game script stats per team-week.

        Phase 8A optimization: Stores cumulative team game script stats for O(1) lookups.
        Eliminates 8-10 redundant PBP filter operations in _extract_game_script_features().

        Args:
            year: Season year
            pbp_df: Pre-loaded PBP DataFrame for the year

        Returns:
            Dict mapping (team, week) -> {
                'team_avg_margin': float,
                'opp_def_ppg_allowed': float,
                'opp_def_ypg_allowed': float,
                'team_plays_per_game': float,
                'team_time_of_possession': float
            }
        """
        if pbp_df is None or len(pbp_df) == 0:
            return {}

        try:
            script_cache = {}
            max_week = pbp_df['week'].max()

            for current_week in range(1, max_week + 1):
                # Filter to data THROUGH current week (cumulative)
                pbp_through_week = pbp_df.filter(pl.col('week') <= current_week)

                # === TEAM SCORING MARGIN ===
                # Get final score for each game
                game_scores = pbp_through_week.group_by(['game_id', 'posteam', 'defteam']).agg([
                    pl.col('total_home_score').last().alias('home_score'),
                    pl.col('total_away_score').last().alias('away_score'),
                    pl.col('home_team').first().alias('home_team')
                ])

                # Calculate margin for each team
                team_margins = []
                for game in game_scores.iter_rows(named=True):
                    home_score = game['home_score'] if game['home_score'] is not None else 0
                    away_score = game['away_score'] if game['away_score'] is not None else 0
                    posteam = game['posteam']
                    home_team = game['home_team']

                    if posteam == home_team:
                        margin = home_score - away_score
                    else:
                        margin = away_score - home_score

                    team_margins.append({'team': posteam, 'margin': margin})

                if len(team_margins) > 0:
                    margins_df = pl.DataFrame(team_margins)
                    avg_margins = margins_df.group_by('team').agg([
                        pl.col('margin').mean().alias('team_avg_margin')
                    ])

                    for row in avg_margins.iter_rows(named=True):
                        team = row['team']
                        cache_key = (team, current_week)

                        if cache_key not in script_cache:
                            script_cache[cache_key] = {}

                        script_cache[cache_key]['team_avg_margin'] = row['team_avg_margin']

                # === OPPONENT DEFENSE PPG & YPG ALLOWED ===
                # For each team, calculate what their opponents allowed (defensive performance)
                game_def_stats = pbp_through_week.group_by(['game_id', 'defteam']).agg([
                    pl.col('total_home_score').last().alias('home_score'),
                    pl.col('total_away_score').last().alias('away_score'),
                    pl.col('home_team').first().alias('home_team'),
                    pl.col('total_home_pass_epa').last().alias('home_pass_epa'),
                    pl.col('total_away_pass_epa').last().alias('away_pass_epa'),
                    pl.col('total_home_rush_epa').last().alias('home_rush_epa'),
                    pl.col('total_away_rush_epa').last().alias('away_rush_epa')
                ])

                # Calculate points allowed per defense
                def_ppg_data = []
                def_ypg_data = []
                for game in game_def_stats.iter_rows(named=True):
                    defteam = game['defteam']
                    home_team = game['home_team']
                    home_score = game['home_score'] if game['home_score'] is not None else 0
                    away_score = game['away_score'] if game['away_score'] is not None else 0

                    # Points allowed by defense
                    if defteam == home_team:
                        points_allowed = away_score
                    else:
                        points_allowed = home_score

                    def_ppg_data.append({'defteam': defteam, 'points_allowed': points_allowed})

                if len(def_ppg_data) > 0:
                    def_ppg_df = pl.DataFrame(def_ppg_data)
                    avg_def_ppg = def_ppg_df.group_by('defteam').agg([
                        pl.col('points_allowed').mean().alias('opp_def_ppg_allowed')
                    ])

                    # For each team, find the average PPG allowed by their opponents
                    # (This requires mapping team -> opponents -> avg defense PPG)
                    # Get team-opponent matchups
                    team_opponents = pbp_through_week.select([
                        'posteam', 'defteam'
                    ]).unique()

                    # Join to get opponent defense stats
                    team_opp_def = team_opponents.join(
                        avg_def_ppg,
                        left_on='defteam',
                        right_on='defteam',
                        how='left'
                    )

                    # Average across all opponents for each team
                    team_avg_opp_def = team_opp_def.group_by('posteam').agg([
                        pl.col('opp_def_ppg_allowed').mean().alias('opp_def_ppg_allowed')
                    ])

                    for row in team_avg_opp_def.iter_rows(named=True):
                        team = row['posteam']
                        cache_key = (team, current_week)

                        if cache_key not in script_cache:
                            script_cache[cache_key] = {}

                        script_cache[cache_key]['opp_def_ppg_allowed'] = row['opp_def_ppg_allowed'] if row['opp_def_ppg_allowed'] is not None else 21.0

                # === OPPONENT YPG ALLOWED (for QB props) ===
                # Calculate total yards allowed by each defense
                def_yards_data = []
                for game_id in pbp_through_week['game_id'].unique():
                    game_plays = pbp_through_week.filter(pl.col('game_id') == game_id)

                    # Get defenses in this game
                    for defteam in game_plays['defteam'].unique():
                        if defteam is None:
                            continue

                        # Yards allowed by this defense in this game
                        def_plays = game_plays.filter(pl.col('defteam') == defteam)
                        total_yards = def_plays['yards_gained'].fill_null(0).sum()

                        def_yards_data.append({'defteam': defteam, 'yards_allowed': total_yards})

                if len(def_yards_data) > 0:
                    def_ypg_df = pl.DataFrame(def_yards_data)
                    avg_def_ypg = def_ypg_df.group_by('defteam').agg([
                        pl.col('yards_allowed').mean().alias('ypg_allowed')
                    ])

                    # Map to teams (same as PPG)
                    team_opp_ypg = team_opponents.join(
                        avg_def_ypg,
                        left_on='defteam',
                        right_on='defteam',
                        how='left'
                    )

                    team_avg_opp_ypg = team_opp_ypg.group_by('posteam').agg([
                        pl.col('ypg_allowed').mean().alias('opp_def_ypg_allowed')
                    ])

                    for row in team_avg_opp_ypg.iter_rows(named=True):
                        team = row['posteam']
                        cache_key = (team, current_week)

                        if cache_key not in script_cache:
                            script_cache[cache_key] = {}

                        script_cache[cache_key]['opp_def_ypg_allowed'] = row['opp_def_ypg_allowed'] if row['opp_def_ypg_allowed'] is not None else 330.0

                # === TEAM PLAYS PER GAME ===
                team_plays = pbp_through_week.filter(
                    (pl.col('play_type').is_in(['pass', 'run'])) &
                    pl.col('posteam').is_not_null()
                ).group_by(['game_id', 'posteam']).agg([
                    pl.count().alias('plays')
                ])

                avg_team_plays = team_plays.group_by('posteam').agg([
                    pl.col('plays').mean().alias('team_plays_per_game')
                ])

                for row in avg_team_plays.iter_rows(named=True):
                    team = row['posteam']
                    cache_key = (team, current_week)

                    if cache_key not in script_cache:
                        script_cache[cache_key] = {}

                    script_cache[cache_key]['team_plays_per_game'] = row['team_plays_per_game']

                # === TEAM TIME OF POSSESSION ===
                # Calculate TOP from play durations
                if 'game_seconds_remaining' in pbp_through_week.columns:
                    # Group by game and team to calculate possession time
                    team_possession = pbp_through_week.filter(
                        pl.col('posteam').is_not_null()
                    ).group_by(['game_id', 'posteam']).agg([
                        pl.count().alias('play_count')  # Proxy for possession
                    ])

                    avg_possession = team_possession.group_by('posteam').agg([
                        pl.col('play_count').mean().alias('team_time_of_possession')
                    ])

                    for row in avg_possession.iter_rows(named=True):
                        team = row['posteam']
                        cache_key = (team, current_week)

                        if cache_key not in script_cache:
                            script_cache[cache_key] = {}

                        # Normalize to 0-1 scale (typical range 55-75 plays)
                        script_cache[cache_key]['team_time_of_possession'] = row['team_time_of_possession'] / 65.0 if row['team_time_of_possession'] else 1.0

            logger.debug(f"  Built game script cache for {year}: {len(script_cache)} team-week entries")
            return script_cache

        except Exception as e:
            logger.warning(f"Error building game script cache for {year}: {e}")
            return {}

    def _build_opponent_defense_cache(self, year: int, pbp_df: Optional[pl.DataFrame]) -> dict:
        """
        Pre-compute CUMULATIVE opponent defense stats per opponent-week.

        Phase 8B optimization: Stores cumulative defensive stats for O(1) lookups.
        Eliminates 2 redundant PBP filter operations in _extract_opponent_defense_features().

        Args:
            year: Season year
            pbp_df: Pre-loaded PBP DataFrame for the year

        Returns:
            Dict mapping (opponent, week) -> {
                'opp_def_pass_ypa': float,
                'opp_def_pass_td_rate': float,
                'opp_def_avg_depth_allowed': float,
                'opp_def_rush_ypc': float,
                'opp_def_rush_td_rate': float
            }
        """
        if pbp_df is None or len(pbp_df) == 0:
            return {}

        try:
            defense_cache = {}
            max_week = pbp_df['week'].max()

            for current_week in range(1, max_week + 1):
                # Filter to data THROUGH current week (cumulative)
                pbp_through_week = pbp_df.filter(pl.col('week') <= current_week)

                # === PASS DEFENSE ===
                pass_attempts = pbp_through_week.filter(
                    (pl.col('pass_attempt') == 1) &
                    pl.col('defteam').is_not_null()
                )

                if len(pass_attempts) > 0:
                    pass_defense = pass_attempts.group_by('defteam').agg([
                        pl.col('passing_yards').fill_null(0).sum().alias('total_pass_yards'),
                        pl.count().alias('pass_attempts'),
                        pl.col('pass_touchdown').fill_null(0).sum().alias('pass_tds_allowed'),
                        pl.col('air_yards').drop_nulls().mean().alias('avg_depth_allowed')
                    ])

                    for row in pass_defense.iter_rows(named=True):
                        opponent = row['defteam']
                        cache_key = (opponent, current_week)

                        if cache_key not in defense_cache:
                            defense_cache[cache_key] = {}

                        attempts = row['pass_attempts']
                        yards = row['total_pass_yards']
                        tds = row['pass_tds_allowed']
                        avg_depth = row.get('avg_depth_allowed')

                        defense_cache[cache_key]['opp_def_pass_ypa'] = yards / attempts if attempts > 0 else 7.0
                        defense_cache[cache_key]['opp_def_pass_td_rate'] = tds / attempts if attempts > 0 else 0.045
                        defense_cache[cache_key]['opp_def_avg_depth_allowed'] = avg_depth if avg_depth is not None else 8.5

                # === RUSH DEFENSE ===
                rush_attempts = pbp_through_week.filter(
                    (pl.col('rush_attempt') == 1) &
                    pl.col('defteam').is_not_null()
                )

                if len(rush_attempts) > 0:
                    rush_defense = rush_attempts.group_by('defteam').agg([
                        pl.col('rushing_yards').fill_null(0).sum().alias('total_rush_yards'),
                        pl.count().alias('rush_attempts'),
                        pl.col('rush_touchdown').fill_null(0).sum().alias('rush_tds_allowed')
                    ])

                    for row in rush_defense.iter_rows(named=True):
                        opponent = row['defteam']
                        cache_key = (opponent, current_week)

                        if cache_key not in defense_cache:
                            defense_cache[cache_key] = {}

                        attempts = row['rush_attempts']
                        yards = row['total_rush_yards']
                        tds = row['rush_tds_allowed']

                        defense_cache[cache_key]['opp_def_rush_ypc'] = yards / attempts if attempts > 0 else 4.3
                        defense_cache[cache_key]['opp_def_rush_td_rate'] = tds / attempts if attempts > 0 else 0.025

            logger.debug(f"  Built opponent defense cache for {year}: {len(defense_cache)} opponent-week entries")
            return defense_cache

        except Exception as e:
            logger.warning(f"Error building opponent defense cache for {year}: {e}")
            return {}

    def _precompute_injury_metrics_batch(
        self,
        year: int,
        multi_year_cache: dict,
        eligible_positions: list[str]
    ) -> dict:
        """
        Batch-compute injury metrics for all players in a year.

        OPTIMIZATION: Pre-compute injury metrics once for all players instead of
        calling count_games_missed_due_to_injury() 4 times per player-week
        (~10,000 N+1 queries eliminated per year).

        Expected speedup: 15-20%

        Args:
            year: Current training year
            multi_year_cache: Cache with injury_data and roster_data for multiple years
            eligible_positions: List of positions being processed (QB, RB, WR, TE)

        Returns:
            Dict mapping (player_id, year) -> {games_played, injury_missed, other_inactive, injury_types}
        """
        import polars as pl

        injury_metrics_batch = {}

        # Compute metrics for current year and 3 prior years (Y, Y-1, Y-2, Y-3)
        # This covers all years needed by _extract_injury_features()
        years_to_compute = [year] + [year - offset for offset in range(1, 4)]

        for compute_year in years_to_compute:
            if compute_year < 2009:  # Injury data starts 2009
                continue

            # Get injury and roster data from multi-year cache
            injury_df = multi_year_cache['injury_data'].get(compute_year)
            roster_df = multi_year_cache['roster_data'].get(compute_year)

            if injury_df is None or roster_df is None:
                continue

            # Get unique players in this year's roster data
            if roster_df.is_empty():
                continue

            unique_players = roster_df.select('gsis_id').unique()

            for player_row in unique_players.iter_rows(named=True):
                player_id = player_row['gsis_id']

                # Filter to this player (regular season only, weeks 1-17)
                player_rosters = roster_df.filter(
                    (pl.col('gsis_id') == player_id) &
                    (pl.col('week') <= 17)
                )

                if len(player_rosters) == 0:
                    continue

                player_injuries = injury_df.filter(
                    (pl.col('gsis_id') == player_id) &
                    (pl.col('week') <= 17)
                )

                # Join roster and injury data
                combined = player_rosters.join(
                    player_injuries.select(['week', 'report_status', 'report_primary_injury']),
                    on='week',
                    how='left'
                )

                # Count games played
                games_played = len(combined.filter(pl.col('status') == 'ACT'))

                # Count injury-related misses
                injury_missed_df = combined.filter(
                    (pl.col('status') == 'RES') |  # IR always counts as injury
                    ((pl.col('status') == 'INA') & (pl.col('report_status') == 'Out'))
                )
                injury_missed = len(injury_missed_df)

                # Count non-injury inactives
                other_inactive = len(combined.filter(
                    (pl.col('status') == 'INA') &
                    (pl.col('report_status').is_null() | (pl.col('report_status') != 'Out'))
                ))

                # Extract injury types
                injury_types = injury_missed_df.select('report_primary_injury').to_series().to_list()
                injury_types = [i for i in injury_types if i is not None]

                # Store result
                injury_metrics_batch[(player_id, compute_year)] = {
                    'games_played': games_played,
                    'injury_missed': injury_missed,
                    'other_inactive': other_inactive,
                    'injury_types': injury_types
                }

        return injury_metrics_batch

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
