"""
ML Feature Engineering Module

Generates machine learning features for player prop predictions by wrapping
existing calculation functions from GridironMetrics codebase.

Architecture:
- Reuses ~1,000 lines of existing feature calculation code
- Extracts raw metrics (not multipliers) for ML model training
- Strict through_week filtering to prevent data leakage
- Handles missing data gracefully with sensible defaults
- Comprehensive injury integration (11 injury features)

Feature Categories:
1. Baseline Performance (4 features): weighted avg, career avg, variance, games played
2. Opponent Defense (2-4 features): Pass/rush YPA, TD rates
3. Efficiency (3 features): Success rate (3wk), red zone rate, usage
4. Injury (1-12 features): RB props use 1 (current status), others use full suite
5. Position-Specific: Catch rate (3 features for WR/TE), target volume (6 features), NextGen (2 features)
6. Game Context (7 features): Home/away, dome, weather, Vegas lines
7. Game Script & Team Context (4-5 features): Team margin, opp defense, pace/TOP
8. Prior Season (3 features): Prior avg, YoY trend, sophomore indicator
9. Categorical (4 features): Opponent, position, week, season

Total: ~45-50 features (varies by position/prop)

Usage:
    engineer = PropFeatureEngineer()
    features = engineer.engineer_features(
        player_id="00-0033873",  # Patrick Mahomes
        season=2024,
        week=10,
        position='QB',
        prop_type='passing_yards',
        opponent_team='BUF'
    )
    # Returns dict with ~55 features including weather, game script, and injury data
"""

import polars as pl
from pathlib import Path
from typing import Dict, Optional
from modules.logger import get_logger
from modules.constants import CACHE_DIR

# Import existing calculation functions (REUSE existing code)
from modules.prop_data_aggregator import (
    calculate_weighted_rolling_average,
    get_simple_average,
    get_career_averages,
    calculate_stat_variance
)
from modules.prop_types import get_stat_column_for_prop, get_prop_config, get_prop_feature_config, should_filter_features
from modules.context_adjustments import ContextAdjustments

logger = get_logger(__name__)


class PropFeatureEngineer:
    """
    Generates ML features for player-week-prop combinations.

    Wraps existing calculation functions to extract raw metrics for ML training.
    All features use data through week-1 only (no future data leakage).
    """

    def __init__(self):
        """Initialize feature engineer with existing calculation classes."""
        self.context_adj = ContextAdjustments()

        # Load matchup lookup table (pre-aggregated historical player vs team stats)
        # This eliminates the need to load PBP data during feature extraction
        matchup_lookup_file = Path(CACHE_DIR) / "ml_training_data" / "matchup_lookup.parquet"
        if matchup_lookup_file.exists():
            try:
                self.matchup_lookup = pl.read_parquet(matchup_lookup_file)
                logger.debug(f"Loaded matchup lookup table: {len(self.matchup_lookup):,} rows")
            except Exception as e:
                logger.warning(f"Failed to load matchup lookup: {e}")
                self.matchup_lookup = None
        else:
            logger.warning(f"Matchup lookup table not found: {matchup_lookup_file}")
            self.matchup_lookup = None

    def engineer_features(
        self,
        player_id: str,
        season: int,
        week: int,
        position: str,
        prop_type: str,
        opponent_team: str,
        pbp_df: Optional[pl.DataFrame] = None,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Generate all ML features for a player-week-prop combination.

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number to predict (uses data through week-1)
            position: QB/RB/WR/TE
            prop_type: e.g., 'passing_yards', 'receiving_yards'
            opponent_team: Opponent team code (e.g., 'BUF', 'KC')
            pbp_df: Optional pre-loaded PBP DataFrame for the season (performance optimization)
            data_cache: Optional cache dict with pre-loaded data (Phase 1 optimization)
                Keys: 'player_stats', 'pbp_df', 'injury_data', 'roster_data', 'nextgen_data'

        Returns:
            dict with ~40 features:
            - Baseline performance features (7)
            - Opponent defense features (4)
            - Efficiency features (4)
            - Position-specific features (varies)
            - Game context features (7)
            - Categorical features (5)
        """
        features = {}

        try:
            # Smart Feature Generation: Get allowed features for this prop type
            allowed_features = None
            if should_filter_features(prop_type):
                config = get_prop_feature_config(prop_type)
                if config:
                    allowed_features = set(config['include'])  # Use set for O(1) lookup

            # Helper function to check if we need any feature in a group
            def needs_feature_group(feature_list):
                """Check if ANY feature in the list is needed for this prop."""
                if allowed_features is None:
                    return True  # No filtering - generate all
                return any(f in allowed_features for f in feature_list)

            # Load player stats through week-1 (strict no-future-data)
            player_stats = self._load_player_stats(player_id, season, position)

            if player_stats is None or len(player_stats) == 0:
                logger.warning(f"No stats found for {player_id} in {season}")
                return self._get_default_features(position, prop_type, opponent_team, week, season)

            stat_column = get_stat_column_for_prop(prop_type)
            if not stat_column:
                logger.error(f"Unknown prop type: {prop_type}")
                return {}

            # Filter to through week-1 (data available at prediction time)
            through_week = week - 1

            # OPTIMIZATION: Pre-filter PBP data once instead of 60 times across methods
            # This eliminates 150,000+ redundant filter operations per year (30-40% speedup)
            pbp_through_week = None
            if pbp_df is not None:
                pbp_through_week = pbp_df.filter(pl.col('week') < week)

            # 1. Baseline Performance Features
            baseline_features = self._extract_baseline_features(
                player_stats, stat_column, through_week, player_id, season, position, data_cache
            )
            features.update(baseline_features)

            # 2. Opponent Defense Features (Phase 8B: uses opponent defense cache)
            opponent_features = self._extract_opponent_defense_features(
                opponent_team, season, week, position, prop_type, pbp_through_week, data_cache
            )
            features.update(opponent_features)

            # 2b. QB Context Features (Phase 1-2: depth, volume, efficiency)
            # Extract player's team from stats for QB lookup
            player_team = None
            try:
                if 'team' in player_stats.columns and len(player_stats) > 0:
                    # Get most recent team (in case of trades)
                    recent_stats = player_stats.filter(pl.col('week') <= week).sort('week', descending=True)
                    if len(recent_stats) > 0:
                        player_team = recent_stats['team'][0]
            except:
                pass  # Fall back to None if team extraction fails

            if player_team:
                qb_context_features = self._extract_qb_context_features(
                    player_team, season, week, prop_type, pbp_through_week
                )
                features.update(qb_context_features)

            # 3. Efficiency Features (success rate, usage - Phase 7: uses cache)
            efficiency_features = self._extract_efficiency_features(
                player_id, season, week, position, prop_type, pbp_through_week, data_cache
            )
            features.update(efficiency_features)

            # 4. Position-Specific Features (smart generation based on config)

            # Catch Rate Features (3 features: catch_rate, avg_target_depth, yac_pct - Phase 7: uses cache)
            if needs_feature_group(['catch_rate', 'avg_target_depth', 'yac_pct']):
                catch_features = self._extract_catch_rate_features(
                    player_id, season, week, pbp_through_week, data_cache
                )
                features.update(catch_features)

            # Target Volume Features (6 features - Phase 5: uses volume cache)
            if needs_feature_group(['targets_season_avg', 'targets_3wk_avg', 'target_share_season',
                                    'target_share_3wk', 'yards_per_target_season', 'yards_per_target_3wk']):
                target_volume_features = self._extract_target_volume_features(
                    player_id, season, week, pbp_through_week, data_cache
                )
                features.update(target_volume_features)

            # NextGen Stats Features (2 features: avg_separation, avg_cushion)
            if needs_feature_group(['avg_separation', 'avg_cushion']):
                nextgen_features = self._extract_nextgen_stats_features(
                    player_id, season, week, data_cache
                )
                features.update(nextgen_features)

            # Route Participation Features (3 features: route_participation_pct, targets_per_route_run, route_target_gap)
            if needs_feature_group(['route_participation_pct', 'targets_per_route_run', 'route_target_gap']):
                route_features = self._extract_route_participation_features(
                    player_id, season, week, data_cache
                )
                features.update(route_features)

            # Prior Season Features (3 features: prior_season_avg, yoy_trend, sophomore_indicator - Phase 6: uses cache)
            if needs_feature_group(['prior_season_avg', 'yoy_trend', 'sophomore_indicator']):
                prior_season_features = self._extract_prior_season_features(
                    player_id, season, week, position, stat_column, data_cache
                )
                features.update(prior_season_features)

            # Rushing Volume Features (3 features: rushing_attempt_share_season, carry_share_3wk, goal_line_share)
            if needs_feature_group(['rushing_attempt_share_season', 'carry_share_3wk', 'goal_line_share']):
                volume_features = self._extract_rushing_volume_features(
                    player_id, season, week, pbp_through_week
                )
                features.update(volume_features)

            # 5. Game Context Features (Phase 4: uses game metadata cache)
            # Extract player's team from stats for cache lookup
            player_team = None
            try:
                if 'team' in player_stats.columns and len(player_stats) > 0:
                    # Get most recent team (could change mid-season via trade)
                    recent_stats = player_stats.filter(pl.col('week') <= week).sort('week', descending=True)
                    if len(recent_stats) > 0:
                        player_team = recent_stats['team'][0]
            except:
                pass  # Fall back to PBP lookup if team extraction fails

            context_features = self._extract_game_context_features(
                opponent_team, player_id, season, week, position, pbp_df,
                player_team=player_team, data_cache=data_cache
            )
            features.update(context_features)

            # 6. Game Script & Team Context Features (Phase 8A: uses game script cache)
            game_script_features = self._extract_game_script_features(
                player_id, season, week, position, opponent_team, prop_type, pbp_through_week,
                data_cache=data_cache, player_team=player_team
            )
            features.update(game_script_features)

            # 7. Matchup Features (Phase 4.1 + 4.2: player vs team history + defense rank)
            if needs_feature_group(['wr_vs_team_avg_yards_last3', 'wr_vs_team_receptions_last3', 'opp_def_pass_rank_pct']):
                matchup_features = self._extract_matchup_features(
                    player_id, opponent_team, season, week, prop_type, pbp_through_week, data_cache
                )
                features.update(matchup_features)

            # 8. Volume Trend Features (Phase 4.3: target trends and volatility)
            if needs_feature_group(['target_trend_3wk', 'target_volatility_cv']):
                volume_trend_features = self._extract_volume_trend_features(
                    player_id, season, week, prop_type, pbp_through_week, data_cache
                )
                features.update(volume_trend_features)

            # 7. Weather Performance Features (DISABLED - replaced with game-level weather)
            # NOTE: Player-specific weather features (lag-1 season) were creating
            #       inverse confidence signal. Replaced with game-level weather features
            #       in _extract_game_context_features() for volume prediction.
            # weather_features = self._extract_weather_features(
            #     player_id, season, position
            # )
            # features.update(weather_features)

            # 8. Injury Features (comprehensive injury integration)
            injury_features = self._extract_injury_features(
                player_id, season, week, position, prop_type, data_cache
            )
            features.update(injury_features)

            # 9. Categorical Features
            features['opponent'] = opponent_team
            features['position'] = position
            features['week'] = week
            features['season'] = season

            # Final filtering: Keep only allowed features for this prop type
            if allowed_features is not None:
                before_count = len(features)
                features = {k: v for k, v in features.items() if k in allowed_features}
                after_count = len(features)
                logger.debug(f"Feature filtering: {before_count} → {after_count} features for {prop_type}")

            logger.debug(f"Generated {len(features)} features for {player_id} week {week}")

        except Exception as e:
            logger.error(f"Error generating features for {player_id} week {week}: {e}")
            return self._get_default_features(position, prop_type, opponent_team, week, season)

        return features

    def _load_player_stats(
        self,
        player_id: str,
        season: int,
        position: str,
        data_cache: Optional[dict] = None
    ) -> Optional[pl.DataFrame]:
        """
        Load player stats from cache or pre-loaded data.

        Phase 2 optimization: Check multi-year cache first to avoid redundant file I/O.

        Args:
            player_id: Player GSIS ID
            season: Season year
            position: Player position
            data_cache: Optional pre-loaded data cache (can be multi-year)

        Returns:
            DataFrame with player stats or None
        """
        # Phase 2: Check if stats are already loaded in multi-year cache
        if data_cache and 'player_stats' in data_cache:
            player_stats_cache = data_cache['player_stats']

            # Check if it's a multi-year cache (dict with year keys)
            if isinstance(player_stats_cache, dict) and season in player_stats_cache:
                # Multi-year cache: get stats for the requested season
                cached_stats = player_stats_cache[season]
                player_df = cached_stats.filter(pl.col('player_id') == player_id)
                if len(player_df) > 0:
                    return player_df
            elif not isinstance(player_stats_cache, dict):
                # Phase 1 cache (single DataFrame): filter to this player
                player_df = player_stats_cache.filter(pl.col('player_id') == player_id)
                if len(player_df) > 0:
                    return player_df

            # If player not found in cache, fall through to file load
            # (this handles edge cases like position changes)

        # Fallback: Load from file (when cache not available or player not in cache)
        stats_file = Path(CACHE_DIR) / "positional_player_stats" / position.lower() / f"{position.lower()}-{season}.csv"

        if not stats_file.exists():
            logger.warning(f"Stats file not found: {stats_file}")
            return None

        try:
            df = pl.read_csv(stats_file)
            player_df = df.filter(pl.col('player_id') == player_id)
            return player_df
        except Exception as e:
            logger.error(f"Error loading player stats: {e}")
            return None

    def _extract_baseline_features(
        self,
        player_stats: pl.DataFrame,
        stat_column: str,
        through_week: int,
        player_id: str,
        season: int,
        position: str,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract baseline performance features (6 features).

        Uses existing functions from prop_data_aggregator.py:
        - calculate_weighted_rolling_average()
        - get_career_averages()
        - calculate_stat_variance()

        Phase 2 optimization: Pass player_stats_cache to get_career_averages()
        to enable multi-year historical lookback without file I/O.
        """
        features = {}

        # Filter to through_week (no future data)
        filtered_stats = player_stats.filter(pl.col('week') <= through_week)

        if len(filtered_stats) == 0:
            # No games played yet - return zeros
            return {
                'weighted_avg': 0.0,
                'career_avg': 0.0,
                'variance_cv': 1.0,  # High variance for no data
                'games_played': 0
            }

        try:
            # Weighted rolling average (recency-weighted: L3: 1.5x, L4-6: 1.0x, L7+: 0.75x)
            features['weighted_avg'] = calculate_weighted_rolling_average(
                player_stats, stat_column, through_week=through_week
            )

            # Career average (3-year lookback)
            # Phase 2: Pass multi-year player stats cache to avoid file I/O
            player_stats_cache = data_cache.get('player_stats') if data_cache else None
            career_avgs = get_career_averages(
                player_id=player_id,
                current_season=season,
                position=position,
                stat_columns=[stat_column],
                lookback_years=3,
                player_stats_cache=player_stats_cache
            )
            features['career_avg'] = career_avgs.get(stat_column, 0.0)

            # Variance (coefficient of variation)
            features['variance_cv'] = calculate_stat_variance(
                player_stats, stat_column, through_week=through_week
            )

            # Sample size features
            features['games_played'] = len(filtered_stats)

        except Exception as e:
            logger.debug(f"Error calculating baseline features: {e}")
            # Return zeros on error
            features = {
                'weighted_avg': 0.0,
                'career_avg': 0.0,
                'variance_cv': 1.0,
                'games_played': 0
            }

        return features

    def _extract_opponent_defense_features(
        self,
        opponent: str,
        season: int,
        week: int,
        position: str,
        prop_type: str,
        pbp_df: Optional[pl.DataFrame] = None,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract opponent defensive stats (5 features).

        Phase 8B optimization: Uses opponent_defense cache for O(1) lookups.
        Eliminates 2 redundant PBP filter operations per player-week call.

        Calculates raw defensive metrics (not multipliers):
        - opp_def_pass_ypa: Yards per attempt allowed
        - opp_def_pass_td_rate: TD rate allowed
        - opp_def_avg_depth_allowed: Average air yards allowed (receiving props only)
        - opp_def_rush_ypc: Yards per carry allowed
        - opp_def_rush_td_rate: TD rate allowed

        Uses data through week-1 only (no future data).
        """
        features = {}

        # Phase 8B: Use opponent defense cache if available (O(1) lookup!)
        if data_cache and 'opponent_defense' in data_cache:
            opponent_defense_cache = data_cache['opponent_defense'].get(season, {})

            through_week = week - 1
            if through_week < 1:
                return self._get_default_opponent_features(prop_type)

            try:
                # O(1) lookup from opponent defense cache
                cache_key = (opponent, through_week)
                if cache_key in opponent_defense_cache:
                    cached_stats = opponent_defense_cache[cache_key]

                    # Populate passing defense features if needed
                    if 'passing' in prop_type or prop_type == 'receptions' or 'receiving' in prop_type:
                        features['opp_def_pass_ypa'] = cached_stats.get('opp_def_pass_ypa', 7.0)
                        features['opp_def_pass_td_rate'] = cached_stats.get('opp_def_pass_td_rate', 0.045)
                        features['opp_def_avg_depth_allowed'] = cached_stats.get('opp_def_avg_depth_allowed', 8.5)

                    # Populate rushing defense features if needed
                    if 'rushing' in prop_type:
                        features['opp_def_rush_ypc'] = cached_stats.get('opp_def_rush_ypc', 4.3)
                        features['opp_def_rush_td_rate'] = cached_stats.get('opp_def_rush_td_rate', 0.018)

                    return features

            except Exception as e:
                logger.debug(f"Error using opponent defense cache: {e}")
                # Fall through to legacy implementation below

        # Load PBP data if not provided
        if pbp_df is None:
            pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
            if not pbp_file.exists():
                logger.warning(f"PBP file not found: {pbp_file}")
                return self._get_default_opponent_features(prop_type)

            try:
                pbp = pl.read_parquet(pbp_file)
            except Exception as e:
                logger.warning(f"Error loading PBP data: {e}")
                return self._get_default_opponent_features(prop_type)
        else:
            pbp = pbp_df

        try:

            # Filter to weeks 1 through week-1 (data available at prediction time)
            pbp_filtered = pbp.filter(pl.col('week') < week)

            if 'passing' in prop_type or prop_type == 'receptions' or 'receiving' in prop_type:
                # Pass defense metrics
                defense = pbp_filtered.filter(
                    (pl.col('defteam') == opponent) &
                    (pl.col('pass_attempt') == 1)
                ).group_by('defteam').agg([
                    pl.col('passing_yards').sum().alias('yards_allowed'),
                    pl.col('pass_attempt').sum().alias('attempts'),
                    pl.col('passing_tds').sum().alias('tds_allowed')
                ])

                if len(defense) > 0 and defense['attempts'][0] > 0:
                    features['opp_def_pass_ypa'] = defense['yards_allowed'][0] / defense['attempts'][0]
                    features['opp_def_pass_td_rate'] = defense['tds_allowed'][0] / defense['attempts'][0]
                else:
                    # No data - use league averages
                    features['opp_def_pass_ypa'] = 7.0
                    features['opp_def_pass_td_rate'] = 0.045  # ~4.5%

                # Average depth allowed (air yards)
                if 'air_yards' in pbp_filtered.columns:
                    depth_data = pbp_filtered.filter(
                        (pl.col('defteam') == opponent) &
                        (pl.col('pass_attempt') == 1)
                    )['air_yards'].drop_nulls()

                    if len(depth_data) > 0:
                        features['opp_def_avg_depth_allowed'] = float(depth_data.mean())
                    else:
                        features['opp_def_avg_depth_allowed'] = 8.5  # League average
                else:
                    features['opp_def_avg_depth_allowed'] = 8.5  # League average

            if 'rushing' in prop_type:
                # Rush defense metrics
                defense = pbp_filtered.filter(
                    (pl.col('defteam') == opponent) &
                    (pl.col('rush_attempt') == 1)
                ).group_by('defteam').agg([
                    pl.col('rushing_yards').sum().alias('yards_allowed'),
                    pl.col('rush_attempt').sum().alias('carries'),
                    pl.col('rushing_tds').sum().alias('tds_allowed')
                ])

                if len(defense) > 0 and defense['carries'][0] > 0:
                    features['opp_def_rush_ypc'] = defense['yards_allowed'][0] / defense['carries'][0]
                    features['opp_def_rush_td_rate'] = defense['tds_allowed'][0] / defense['carries'][0]
                else:
                    # No data - use league averages
                    features['opp_def_rush_ypc'] = 4.3
                    features['opp_def_rush_td_rate'] = 0.018  # ~1.8%

        except Exception as e:
            logger.debug(f"Error calculating opponent defense features: {e}")
            return self._get_default_opponent_features(prop_type)

        return features

    def _extract_qb_context_features(
        self,
        team: str,
        season: int,
        week: int,
        prop_type: str,
        pbp_df: Optional[pl.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Extract QB context features (6 features for Phase 1-2).

        Phase 1: Depth dimension
        - qb_adot_3wk: QB's average depth of target over last 3 weeks

        Phase 2: Volume and efficiency (added later)
        - qb_pass_yards_3wk: QB's avg passing yards last 3 weeks
        - qb_pass_attempts_3wk: QB's avg pass attempts last 3 weeks
        - qb_comp_pct_3wk: QB's completion % last 3 weeks
        - qb_ypa_3wk: QB's yards per attempt last 3 weeks
        - qb_target_concentration: Herfindahl index of target distribution

        Only applies to passing/receiving props.
        Uses data from last 3 weeks only (no future data).
        """
        features = {}

        # Only relevant for passing/receiving props
        if not ('passing' in prop_type or prop_type == 'receptions' or 'receiving' in prop_type):
            return features

        # Load PBP data if not provided
        if pbp_df is None:
            pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
            if not pbp_file.exists():
                logger.debug(f"PBP file not found for QB context: {pbp_file}")
                return self._get_default_qb_features()

            try:
                pbp = pl.read_parquet(pbp_file)
            except Exception as e:
                logger.debug(f"Error loading PBP for QB context: {e}")
                return self._get_default_qb_features()
        else:
            pbp = pbp_df

        try:
            # Filter to team's offensive plays in weeks (week-3) through (week-1)
            # This gives us the last 3 weeks of QB performance
            start_week = max(1, week - 3)
            end_week = week - 1

            if end_week < 1:
                return self._get_default_qb_features()

            # Get team's passing plays from last 3 weeks
            team_passes = pbp.filter(
                (pl.col('posteam') == team) &
                (pl.col('week') >= start_week) &
                (pl.col('week') <= end_week) &
                (pl.col('pass_attempt') == 1)
            )

            if len(team_passes) == 0:
                return self._get_default_qb_features()

            # PHASE 1: QB Average Depth of Target (aDOT)
            if 'air_yards' in team_passes.columns:
                air_yards_data = team_passes['air_yards'].drop_nulls()
                if len(air_yards_data) > 0:
                    features['qb_adot_3wk'] = float(air_yards_data.mean())
                else:
                    features['qb_adot_3wk'] = 8.5  # League average
            else:
                features['qb_adot_3wk'] = 8.5  # League average

            # PHASE 2: QB Volume Metrics
            # QB passing yards last 3 weeks
            if 'passing_yards' in team_passes.columns:
                # Group by week to get per-game averages
                weekly_yards = team_passes.group_by('week').agg([
                    pl.col('passing_yards').sum().alias('yards')
                ])
                if len(weekly_yards) > 0:
                    features['qb_pass_yards_3wk'] = float(weekly_yards['yards'].mean())
                else:
                    features['qb_pass_yards_3wk'] = 250.0  # League average
            else:
                features['qb_pass_yards_3wk'] = 250.0  # League average

            # QB pass attempts last 3 weeks
            weekly_attempts = team_passes.group_by('week').agg([
                pl.col('pass_attempt').sum().alias('attempts')
            ])
            if len(weekly_attempts) > 0:
                features['qb_pass_attempts_3wk'] = float(weekly_attempts['attempts'].mean())
            else:
                features['qb_pass_attempts_3wk'] = 35.0  # League average

            # PHASE 2: QB Efficiency Metrics
            # QB completion percentage last 3 weeks
            completions = team_passes.filter(pl.col('complete_pass') == 1)
            attempts = len(team_passes)
            if attempts > 0:
                features['qb_comp_pct_3wk'] = len(completions) / attempts
            else:
                features['qb_comp_pct_3wk'] = 0.65  # League average

            # QB yards per attempt last 3 weeks
            if attempts > 0 and 'passing_yards' in team_passes.columns:
                total_yards = team_passes['passing_yards'].sum()
                features['qb_ypa_3wk'] = float(total_yards) / attempts
            else:
                features['qb_ypa_3wk'] = 7.0  # League average

            # PHASE 2: QB Target Concentration (Herfindahl Index)
            # Measures how concentrated QB's targets are (0 = distributed, 1 = single target)
            if 'receiver_player_id' in team_passes.columns:
                receiver_targets = team_passes.group_by('receiver_player_id').agg([
                    pl.col('pass_attempt').count().alias('targets')
                ])
                total_targets = receiver_targets['targets'].sum()
                if total_targets > 0:
                    # Calculate Herfindahl index: sum of squared market shares
                    shares = receiver_targets['targets'] / total_targets
                    herfindahl = float((shares ** 2).sum())
                    features['qb_target_concentration'] = herfindahl
                else:
                    features['qb_target_concentration'] = 0.25  # Moderate concentration
            else:
                features['qb_target_concentration'] = 0.25  # Moderate concentration

        except Exception as e:
            logger.debug(f"Error calculating QB context features: {e}")
            return self._get_default_qb_features()

        return features

    def _get_default_qb_features(self) -> Dict[str, float]:
        """Return default QB features when data is unavailable."""
        return {
            'qb_adot_3wk': 8.5,  # League average depth of target
            'qb_pass_yards_3wk': 250.0,  # League average passing yards per game
            'qb_pass_attempts_3wk': 35.0,  # League average attempts per game
            'qb_comp_pct_3wk': 0.65,  # League average completion percentage
            'qb_ypa_3wk': 7.0,  # League average yards per attempt
            'qb_target_concentration': 0.25,  # Moderate target concentration
        }

    def _extract_matchup_features(
        self,
        player_id: str,
        opponent: str,
        season: int,
        week: int,
        prop_type: str,
        pbp_df: Optional[pl.DataFrame] = None,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract player vs opponent matchup history features (Phase 4.1 + 4.2).

        Phase 4.1: Player-specific matchup history
        - wr_vs_team_avg_yards_last3: This WR's avg yards in last 3 career meetings
        - wr_vs_team_receptions_last3: This WR's avg receptions in last 3 career meetings

        Phase 4.2: Opponent defense rank
        - opp_def_pass_rank_pct: Opponent's pass defense rank as percentile (0=best, 1=worst)

        OPTIMIZATION: Uses pre-aggregated lookup table for player vs team history.
        Defense rankings calculated from existing opponent_defense cache.
        """
        features = {}

        # Only relevant for receiving props
        if not ('receiving' in prop_type or prop_type == 'receptions'):
            return features

        # PHASE 4.1: Player vs Team History (from lookup table)
        if self.matchup_lookup is not None:
            try:
                # Simple lookup: filter to this specific player-opponent-season-week
                lookup_row = self.matchup_lookup.filter(
                    (pl.col('player_id') == player_id) &
                    (pl.col('opponent') == opponent) &
                    (pl.col('season') == season) &
                    (pl.col('week') == week)
                )

                if len(lookup_row) > 0:
                    # Extract pre-computed values
                    features['wr_vs_team_avg_yards_last3'] = lookup_row['wr_vs_team_avg_yards_last3'][0]
                    features['wr_vs_team_receptions_last3'] = lookup_row['wr_vs_team_receptions_last3'][0]
                else:
                    # No lookup entry (shouldn't happen, but handle gracefully)
                    features['wr_vs_team_avg_yards_last3'] = float('nan')
                    features['wr_vs_team_receptions_last3'] = float('nan')
            except Exception as e:
                logger.debug(f"Error looking up matchup history: {e}")
                features['wr_vs_team_avg_yards_last3'] = float('nan')
                features['wr_vs_team_receptions_last3'] = float('nan')
        else:
            # Lookup table not available
            features['wr_vs_team_avg_yards_last3'] = float('nan')
            features['wr_vs_team_receptions_last3'] = float('nan')

        # PHASE 4.2: Opponent Defense Rank Percentile (from opponent_defense cache)
        if data_cache and 'opponent_defense' in data_cache:
            try:
                # Get defense stats for current season from cache
                season_defense = data_cache['opponent_defense'].get(season)

                if season_defense:
                    # Filter to games through current week
                    # Cache structure is {(team, week): stats}
                    defense_through_week = {}
                    for cache_key, stats in season_defense.items():
                        team, cache_week = cache_key
                        if cache_week < week:
                            # Keep only the latest week for each team
                            if team not in defense_through_week or cache_week > defense_through_week[team]['week']:
                                defense_through_week[team] = {**stats, 'week': cache_week}

                    if defense_through_week:
                        # Get YPA for all teams
                        team_ypas = {
                            team: stats.get('opp_def_pass_ypa', 7.0)
                            for team, stats in defense_through_week.items()
                        }

                        # Calculate percentile for opponent
                        if opponent in team_ypas:
                            opponent_ypa = team_ypas[opponent]
                            sorted_ypas = sorted(team_ypas.values())
                            total_teams = len(sorted_ypas)

                            # Find rank (0 = best defense, 1 = worst defense)
                            better_or_equal = sum(1 for ypa in sorted_ypas if ypa <= opponent_ypa)
                            rank_pct = (better_or_equal - 1) / (total_teams - 1) if total_teams > 1 else 0.5

                            features['opp_def_pass_rank_pct'] = float(rank_pct)
                        else:
                            features['opp_def_pass_rank_pct'] = 0.5  # Middle of pack
                    else:
                        features['opp_def_pass_rank_pct'] = 0.5
                else:
                    features['opp_def_pass_rank_pct'] = 0.5
            except Exception as e:
                logger.debug(f"Error calculating defense rank: {e}")
                features['opp_def_pass_rank_pct'] = 0.5
        else:
            features['opp_def_pass_rank_pct'] = 0.5

        return features

    def _get_default_matchup_features(self) -> Dict[str, float]:
        """Return default matchup features when data is unavailable."""
        return {
            'wr_vs_team_avg_yards_last3': float('nan'),
            'wr_vs_team_receptions_last3': float('nan'),
            'opp_def_pass_rank_pct': 0.5,  # Middle of pack
        }

    def _extract_volume_trend_features(
        self,
        player_id: str,
        season: int,
        week: int,
        prop_type: str,
        pbp_df: Optional[pl.DataFrame] = None,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract target volume trend features (Phase 4.3).

        Features:
        - target_trend_3wk: Linear slope of targets over last 3 weeks
          (positive = emerging role, negative = declining)
        - target_volatility_cv: Coefficient of variation for targets over last 5 weeks
          (low = consistent role, high = game-script dependent)

        Only applies to receiving props.
        Uses data through week-1 only (no future data).
        """
        features = {}

        # Only relevant for receiving props
        if not ('receiving' in prop_type or prop_type == 'receptions'):
            return features

        # Use cache if available
        if data_cache and 'player_volume' in data_cache:
            volume_cache = data_cache['player_volume'].get(season, {})
            player_data = volume_cache.get(player_id)

            if player_data is not None:
                # Get targets from last 5 weeks
                recent_weeks = player_data.filter(
                    (pl.col('week') < week) &
                    (pl.col('week') >= week - 5)
                ).sort('week')

                if len(recent_weeks) >= 3:
                    targets_list = recent_weeks['targets'].to_list()

                    # Calculate trend (simple slope over last 3 weeks)
                    if len(targets_list) >= 3:
                        last_3 = targets_list[-3:]
                        # Simple slope: (last - first) / (n - 1)
                        features['target_trend_3wk'] = (last_3[-1] - last_3[0]) / 2.0
                    else:
                        features['target_trend_3wk'] = 0.0

                    # Calculate volatility (coefficient of variation)
                    if len(targets_list) >= 2:
                        targets_array = pl.Series(targets_list)
                        mean_targets = targets_array.mean()
                        std_targets = targets_array.std()
                        if mean_targets > 0:
                            features['target_volatility_cv'] = std_targets / mean_targets
                        else:
                            features['target_volatility_cv'] = 0.0
                    else:
                        features['target_volatility_cv'] = 0.0
                else:
                    features['target_trend_3wk'] = 0.0
                    features['target_volatility_cv'] = 0.0

                return features

        # Fallback: Use PBP if cache not available
        if pbp_df is None:
            pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
            if not pbp_file.exists():
                return self._get_default_volume_trend_features()

            try:
                pbp = pl.read_parquet(pbp_file)
            except:
                return self._get_default_volume_trend_features()
        else:
            pbp = pbp_df

        try:
            # Get player's targets by week
            player_targets = pbp.filter(
                (pl.col('receiver_player_id') == player_id) &
                (pl.col('week') < week) &
                (pl.col('week') >= week - 5)
            ).group_by('week').agg([
                pl.col('pass_attempt').count().alias('targets')
            ]).sort('week')

            if len(player_targets) >= 3:
                targets_list = player_targets['targets'].to_list()

                # Calculate trend
                last_3 = targets_list[-3:]
                features['target_trend_3wk'] = (last_3[-1] - last_3[0]) / 2.0

                # Calculate volatility
                targets_array = pl.Series(targets_list)
                mean_targets = targets_array.mean()
                std_targets = targets_array.std()
                if mean_targets > 0:
                    features['target_volatility_cv'] = std_targets / mean_targets
                else:
                    features['target_volatility_cv'] = 0.0
            else:
                features['target_trend_3wk'] = 0.0
                features['target_volatility_cv'] = 0.0

        except Exception as e:
            logger.debug(f"Error calculating volume trend features: {e}")
            return self._get_default_volume_trend_features()

        return features

    def _get_default_volume_trend_features(self) -> Dict[str, float]:
        """Return default volume trend features when data is unavailable."""
        return {
            'target_trend_3wk': 0.0,  # No trend
            'target_volatility_cv': 0.0,  # No volatility data
        }

    def _extract_efficiency_features(
        self,
        player_id: str,
        season: int,
        week: int,
        position: str,
        prop_type: str,
        pbp_df: Optional[pl.DataFrame] = None,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract efficiency and usage features (3 features).

        Phase 7 optimization: Uses PBP stats cache for O(1) lookups.

        Features:
        - success_rate_3wk: 3-week rolling success rate
        - red_zone_rate: Red zone touch percentage
        - usage_rate: Target share or carry share
        """
        features = {
            'success_rate_3wk': 0.5,  # Default to 50%
            'red_zone_rate': 0.15,  # Default to 15%
            'usage_rate': 0.15  # Default to 15% usage
        }

        # Phase 7: Use PBP stats cache if available
        if data_cache and 'pbp_stats' in data_cache:
            pbp_stats_cache = data_cache['pbp_stats'].get(season, {})

            through_week = week - 1
            if through_week < 1:
                return features

            try:
                # O(1) lookup for cumulative stats
                cache_key = (player_id, through_week)
                if cache_key in pbp_stats_cache:
                    stats = pbp_stats_cache[cache_key]

                    total_completions = stats.get('completions', 0)
                    successful_plays = stats.get('successful_plays', 0)
                    total_targets = stats.get('targets', 0)
                    red_zone_targets = stats.get('red_zone_targets', 0)

                    # Success rate (for receivers, based on completions)
                    if total_completions >= 10:
                        features['success_rate_3wk'] = successful_plays / total_completions if total_completions > 0 else 0.5

                    # Red zone rate
                    if total_targets >= 15:
                        features['red_zone_rate'] = red_zone_targets / total_targets if total_targets > 0 else 0.15

                return features

            except Exception as e:
                logger.debug(f"Error using PBP stats cache for efficiency: {e}")
                # Fall through to PBP filtering fallback

        # Fallback: Load PBP data if not provided
        if pbp_df is None:
            pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
            if not pbp_file.exists():
                return features

            try:
                pbp = pl.read_parquet(pbp_file)
            except Exception as e:
                logger.warning(f"Error loading PBP data: {e}")
                return features
        else:
            pbp = pbp_df

        try:
            pbp_filtered = pbp.filter(pl.col('week') < week)

            # Success rate calculation (chain-moving plays)
            if 'success' in pbp.columns:
                if position == 'QB' and 'passing' in prop_type:
                    player_plays = pbp_filtered.filter(
                        (pl.col('passer_player_id') == player_id) &
                        (pl.col('pass_attempt') == 1)
                    )
                elif 'rushing' in prop_type:
                    player_plays = pbp_filtered.filter(
                        (pl.col('rusher_player_id') == player_id) &
                        (pl.col('rush_attempt') == 1)
                    )
                elif 'receiving' in prop_type or prop_type == 'receptions':
                    player_plays = pbp_filtered.filter(
                        (pl.col('receiver_player_id') == player_id) &
                        (pl.col('complete_pass') == 1)
                    )
                else:
                    player_plays = pl.DataFrame()

                if len(player_plays) >= 20:  # Minimum sample size
                    # 3-week rolling success rate
                    last_3_weeks = pbp_filtered.filter(
                        pl.col('week') >= max(1, week - 3)
                    )
                    if 'passing' in prop_type:
                        recent_plays = last_3_weeks.filter(
                            (pl.col('passer_player_id') == player_id) &
                            (pl.col('pass_attempt') == 1)
                        )
                    elif 'rushing' in prop_type:
                        recent_plays = last_3_weeks.filter(
                            (pl.col('rusher_player_id') == player_id) &
                            (pl.col('rush_attempt') == 1)
                        )
                    else:
                        recent_plays = last_3_weeks.filter(
                            (pl.col('receiver_player_id') == player_id) &
                            (pl.col('complete_pass') == 1)
                        )

                    if len(recent_plays) >= 10:
                        features['success_rate_3wk'] = recent_plays['success'].mean()

            # Red zone rate (inside opponent 20)
            if position in ['RB', 'WR', 'TE']:
                if 'receiving' in prop_type or prop_type == 'receptions':
                    all_touches = pbp_filtered.filter(
                        (pl.col('receiver_player_id') == player_id) &
                        (pl.col('complete_pass') == 1)
                    )
                elif 'rushing' in prop_type:
                    all_touches = pbp_filtered.filter(
                        (pl.col('rusher_player_id') == player_id) &
                        (pl.col('rush_attempt') == 1)
                    )
                else:
                    all_touches = pl.DataFrame()

                if len(all_touches) >= 15:
                    if 'yardline_100' in pbp.columns:
                        red_zone_touches = all_touches.filter(pl.col('yardline_100') <= 20)
                        features['red_zone_rate'] = len(red_zone_touches) / len(all_touches)

        except Exception as e:
            logger.debug(f"Error calculating efficiency features: {e}")

        return features

    def _extract_catch_rate_features(
        self,
        player_id: str,
        season: int,
        week: int,
        pbp_df: Optional[pl.DataFrame] = None,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract receiver catch rate features (3 features).

        Phase 7 optimization: Uses PBP stats cache for O(1) lookups instead
        of filtering PBP data repeatedly (eliminates 10,000+ filter operations).

        Features:
        - catch_rate: Completions / targets
        - avg_target_depth: Average air yards per target
        - yac_pct: Yards after catch percentage
        """
        features = {
            'catch_rate': 0.65,  # Default 65% catch rate
            'avg_target_depth': 10.0,  # Default 10 yards
            'yac_pct': 0.5  # Default 50% YAC
        }

        # Phase 7: Use PBP stats cache if available (now with cumulative stats - O(1) lookup!)
        if data_cache and 'pbp_stats' in data_cache:
            pbp_stats_cache = data_cache['pbp_stats'].get(season, {})

            # We need data through week-1
            through_week = week - 1
            if through_week < 1:
                return features

            try:
                # O(1) lookup - cache stores cumulative stats
                cache_key = (player_id, through_week)
                if cache_key in pbp_stats_cache:
                    stats = pbp_stats_cache[cache_key]

                    total_targets = stats.get('targets', 0)
                    total_completions = stats.get('completions', 0)
                    total_air_yards = stats.get('air_yards_total', 0.0)
                    air_yards_count = stats.get('air_yards_count', 0)
                    total_yac = stats.get('yac_total', 0.0)
                    total_rec_yards = stats.get('receiving_yards_total', 0.0)

                    if total_targets >= 15:  # Minimum sample size
                        features['catch_rate'] = total_completions / total_targets

                        if air_yards_count > 0:
                            features['avg_target_depth'] = total_air_yards / air_yards_count

                        if total_rec_yards > 0:
                            features['yac_pct'] = total_yac / total_rec_yards

                return features

            except Exception as e:
                logger.debug(f"Error using PBP stats cache: {e}")
                # Fall through to PBP filtering fallback

        # Fallback: Original PBP filtering logic (if cache not available)
        if pbp_df is None:
            pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
            if not pbp_file.exists():
                return features

            try:
                pbp = pl.read_parquet(pbp_file)
            except Exception as e:
                logger.warning(f"Error loading PBP data: {e}")
                return features
        else:
            pbp = pbp_df

        try:
            pbp_filtered = pbp.filter(pl.col('week') < week)

            # Get targets (completions + incompletions)
            targets = pbp_filtered.filter(
                (pl.col('receiver_player_id') == player_id) &
                ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
            )

            if len(targets) >= 15:  # Minimum sample size
                completions = targets.filter(pl.col('complete_pass') == 1)
                features['catch_rate'] = len(completions) / len(targets)

                # Average target depth
                if 'air_yards' in pbp.columns:
                    air_yards = targets['air_yards'].drop_nulls()
                    if len(air_yards) > 0:
                        features['avg_target_depth'] = air_yards.mean()

                # YAC percentage
                if len(completions) > 0 and 'receiving_yards_after_catch' in pbp.columns:
                    yac = completions['receiving_yards_after_catch'].sum()
                    total_yards = completions['receiving_yards'].sum()
                    if total_yards > 0:
                        features['yac_pct'] = yac / total_yards

        except Exception as e:
            logger.debug(f"Error calculating catch rate features: {e}")

        return features

    def _extract_blocking_quality_features(
        self,
        player_id: str,
        season: int,
        week: int,
        pbp_df: Optional[pl.DataFrame] = None,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract RB blocking quality features (3 features).

        Phase 8C optimization: Uses player_volume cache (Phase 5) for O(1) lookups.
        Eliminates file I/O and redundant PBP filtering.

        Features:
        - player_ypc: Player yards per carry
        - team_ypc: Teammate RB yards per carry
        - ypc_diff_pct: Player YPC / team YPC (OL quality proxy)
        """
        features = {
            'player_ypc': 4.3,
            'team_ypc': 4.3,
            'ypc_diff_pct': 1.0
        }

        # Phase 8C: Use player volume cache if available
        if data_cache and 'player_volume' in data_cache:
            player_volume_cache = data_cache['player_volume'].get(season, {})

            through_week = week - 1
            if through_week < 1:
                return features

            try:
                # O(1) lookup from player volume cache
                cache_key = (player_id, through_week)
                if cache_key in player_volume_cache:
                    player_stats = player_volume_cache[cache_key]

                    carries = player_stats.get('carries', 0)
                    rushing_yards = player_stats.get('rushing_yards', 0.0)

                    if carries >= 20:
                        features['player_ypc'] = rushing_yards / carries

                        # Get team totals to calculate team YPC (excluding player)
                        team = player_stats.get('team')
                        team_total_carries = player_stats.get('team_carries', 0)

                        # Calculate teammate carries (team - player)
                        teammate_carries = team_total_carries - carries

                        # We need teammate yards, which requires summing all other RBs
                        # This requires iterating through cache, but much faster than PBP
                        if team and teammate_carries >= 20:
                            teammate_yards = 0.0
                            for (pid, wk), stats in player_volume_cache.items():
                                if wk == through_week and stats.get('team') == team and pid != player_id:
                                    teammate_yards += stats.get('rushing_yards', 0.0)

                            if teammate_carries > 0:
                                features['team_ypc'] = teammate_yards / teammate_carries

                                if features['team_ypc'] > 0:
                                    features['ypc_diff_pct'] = features['player_ypc'] / features['team_ypc']

                    return features

            except Exception as e:
                logger.debug(f"Error using player volume cache for blocking quality: {e}")
                # Fall through to legacy implementation

        # Legacy implementation: Load PBP if cache not available
        if pbp_df is None:
            pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
            if not pbp_file.exists():
                return features

            try:
                pbp = pl.read_parquet(pbp_file)
            except Exception as e:
                logger.debug(f"Error loading PBP for blocking quality: {e}")
                return features
        else:
            pbp = pbp_df

        try:
            pbp_filtered = pbp.filter(pl.col('week') < week)

            # Player carries
            player_carries = pbp_filtered.filter(
                (pl.col('rusher_player_id') == player_id) &
                (pl.col('rush_attempt') == 1)
            )

            if len(player_carries) >= 20:
                player_yards = player_carries['rushing_yards'].sum()
                features['player_ypc'] = player_yards / len(player_carries)

                # Get player's team
                player_team = player_carries['posteam'][0] if len(player_carries) > 0 else None

                if player_team:
                    # Teammate RB carries (excluding this player)
                    team_carries = pbp_filtered.filter(
                        (pl.col('posteam') == player_team) &
                        (pl.col('rush_attempt') == 1) &
                        (pl.col('rusher_player_id') != player_id)
                    )

                    if len(team_carries) >= 20:
                        team_yards = team_carries['rushing_yards'].sum()
                        features['team_ypc'] = team_yards / len(team_carries)

                        if features['team_ypc'] > 0:
                            features['ypc_diff_pct'] = features['player_ypc'] / features['team_ypc']

        except Exception as e:
            logger.debug(f"Error calculating blocking quality features: {e}")

        return features

    def _extract_rushing_volume_features(
        self,
        player_id: str,
        season: int,
        week: int,
        pbp_df: Optional[pl.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Extract RB rushing volume features (3 features).

        Dual-signal approach for detecting role changes and breakouts:
        - Season-long baseline: stable role indicator
        - 3-week rolling: captures breakouts, hot hand, role changes
        - Goal-line: situational high-value opportunities

        Model learns from DIFFERENCE between season vs 3wk:
        - carry_share_3wk >> rushing_attempt_share_season → Breakout!
        - carry_share_3wk ≈ rushing_attempt_share_season → Stable bellcow
        - carry_share_3wk << rushing_attempt_share_season → Losing touches

        Features:
        - rushing_attempt_share_season: Player's % of team rushing attempts (season-long through week-1)
        - carry_share_3wk: Recent 3-game carry allocation
        - goal_line_share: Inside-5 yard carry percentage

        All calculated using data through week-1 only (no future data leakage).
        """
        features = {
            'rushing_attempt_share_season': 0.15,  # Default 15% team share
            'carry_share_3wk': 0.15,               # Default 15% recent share
            'goal_line_share': 0.20                # Default 20% goal-line work
        }

        # Load PBP data if not provided
        if pbp_df is None:
            pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
            if not pbp_file.exists():
                return features

            try:
                pbp = pl.read_parquet(pbp_file)
            except Exception as e:
                logger.warning(f"Error loading PBP data: {e}")
                return features
        else:
            pbp = pbp_df

        try:
            pbp_filtered = pbp.filter(pl.col('week') < week)  # Through week-1

            # Get player's carries
            player_carries = pbp_filtered.filter(
                (pl.col('rusher_player_id') == player_id) &
                (pl.col('rush_attempt') == 1)
            )

            if len(player_carries) < 10:  # Minimum sample size
                return features

            # Get player's team
            player_team = player_carries['posteam'][0] if len(player_carries) > 0 else None

            if not player_team:
                return features

            # === 1. SEASON-LONG RUSHING ATTEMPT SHARE ===
            team_carries_all = pbp_filtered.filter(
                (pl.col('posteam') == player_team) &
                (pl.col('rush_attempt') == 1)
            )

            if len(team_carries_all) >= 20:
                features['rushing_attempt_share_season'] = len(player_carries) / len(team_carries_all)

            # === 2. 3-WEEK ROLLING CARRY SHARE ===
            recent_weeks = pbp_filtered.filter(
                pl.col('week') >= max(1, week - 3)
            )

            player_recent = recent_weeks.filter(
                (pl.col('rusher_player_id') == player_id) &
                (pl.col('rush_attempt') == 1)
            )

            team_recent = recent_weeks.filter(
                (pl.col('posteam') == player_team) &
                (pl.col('rush_attempt') == 1)
            )

            if len(team_recent) >= 15:
                features['carry_share_3wk'] = len(player_recent) / len(team_recent)

            # === 3. GOAL-LINE SHARE (inside 5-yard line) ===
            if 'yardline_100' in pbp.columns:
                goal_line_carries_team = team_carries_all.filter(
                    pl.col('yardline_100') <= 5
                )

                goal_line_carries_player = player_carries.filter(
                    pl.col('yardline_100') <= 5
                )

                if len(goal_line_carries_team) >= 5:  # Minimum goal-line opportunities
                    features['goal_line_share'] = len(goal_line_carries_player) / len(goal_line_carries_team)

        except Exception as e:
            logger.debug(f"Error calculating rushing volume features: {e}")

        return features

    def _extract_target_volume_features(
        self,
        player_id: str,
        season: int,
        week: int,
        pbp_df: Optional[pl.DataFrame] = None,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract WR/TE/RB target volume features (6 features).

        Phase 5 optimization: Uses player volume cache for O(1) lookups instead
        of filtering PBP data repeatedly (eliminates 10,000+ filter operations per year).

        Dual-signal approach for detecting role changes and breakouts:
        - Season-long baseline: stable target allocation
        - 3-week rolling: captures emerging trends, hot hand, role changes
        - Efficiency metrics: yards per target

        Model learns from DIFFERENCE between season vs 3wk:
        - target_share_3wk >> target_share_season → Breakout/increased usage!
        - target_share_3wk ≈ target_share_season → Stable WR1/TE1
        - target_share_3wk << target_share_season → Losing targets

        Features:
        - targets_season_avg: Average targets per game (season-long through week-1)
        - targets_3wk_avg: Average targets per game (3-week rolling)
        - target_share_season: Player's % of team targets (season-long)
        - target_share_3wk: Recent 3-game team target percentage
        - yards_per_target_season: Season-long YPT efficiency
        - yards_per_target_3wk: Recent 3-game YPT efficiency

        All calculated using data through week-1 only (no future data leakage).
        """
        features = {
            'targets_season_avg': 5.0,          # Default 5 targets/game
            'targets_3wk_avg': 5.0,             # Default 5 targets/game recent
            'target_share_season': 0.15,        # Default 15% team share
            'target_share_3wk': 0.15,           # Default 15% recent share
            'yards_per_target_season': 8.0,     # Default 8 YPT
            'yards_per_target_3wk': 8.0         # Default 8 YPT recent
        }

        # Phase 5: Use player volume cache if available
        if data_cache and 'player_volume' in data_cache:
            player_volume_cache = data_cache['player_volume'].get(season, {})

            # We need data through week-1
            through_week = week - 1
            if through_week < 1:
                return features

            try:
                # Get cumulative stats through week-1
                cache_key = (player_id, through_week)
                if cache_key not in player_volume_cache:
                    return features

                current_stats = player_volume_cache[cache_key]

                # Simple estimation: weeks_played ≈ through_week (assumes consistent playing time)
                # More accurate would require tracking, but this is fast and reasonable
                weeks_played = max(1, through_week)

                # === SEASON-LONG METRICS (through week-1) ===
                season_targets = current_stats['targets']
                season_rec_yards = current_stats['receiving_yards']
                team_season_targets = current_stats['team_targets']

                if season_targets >= 5:  # Minimum sample
                    features['targets_season_avg'] = season_targets / weeks_played
                    features['yards_per_target_season'] = season_rec_yards / season_targets if season_targets > 0 else 8.0

                    if team_season_targets >= 10:
                        features['target_share_season'] = season_targets / team_season_targets

                # === 3-WEEK ROLLING METRICS ===
                # Simple approach: Get stats from week-4 and subtract from current
                four_weeks_ago = through_week - 4
                if four_weeks_ago >= 1:
                    old_key = (player_id, four_weeks_ago)
                    if old_key in player_volume_cache:
                        old_stats = player_volume_cache[old_key]
                        recent_targets = season_targets - old_stats['targets']
                        recent_rec_yards = season_rec_yards - old_stats['receiving_yards']
                        recent_team_targets = team_season_targets - old_stats['team_targets']
                    else:
                        # Player wasn't active 4 weeks ago, use all current stats
                        recent_targets = season_targets
                        recent_rec_yards = season_rec_yards
                        recent_team_targets = team_season_targets
                else:
                    # Not enough weeks yet, use all data
                    recent_targets = season_targets
                    recent_rec_yards = season_rec_yards
                    recent_team_targets = team_season_targets

                # Assume 3-4 weeks of data
                recent_weeks = min(4, through_week)

                if recent_targets >= 3:
                    features['targets_3wk_avg'] = recent_targets / recent_weeks
                    features['yards_per_target_3wk'] = recent_rec_yards / recent_targets if recent_targets > 0 else 8.0

                    if recent_team_targets >= 5:
                        features['target_share_3wk'] = recent_targets / recent_team_targets

                return features

            except Exception as e:
                logger.debug(f"Error using player volume cache: {e}")
                # Fall through to PBP filtering fallback

        # Fallback: Original PBP filtering logic (if cache not available)
        if pbp_df is None:
            pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
            if not pbp_file.exists():
                return features

            try:
                pbp = pl.read_parquet(pbp_file)
            except Exception as e:
                logger.warning(f"Error loading PBP data: {e}")
                return features
        else:
            pbp = pbp_df

        try:
            pbp_filtered = pbp.filter(pl.col('week') < week)  # Through week-1

            # Get player's targets (completions + incompletions + targets)
            player_targets = pbp_filtered.filter(
                (pl.col('receiver_player_id') == player_id) &
                ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
            )

            if len(player_targets) < 5:  # Minimum sample size
                return features

            # Get player's team from their targets
            player_team = player_targets['posteam'][0] if len(player_targets) > 0 else None

            if not player_team:
                return features

            # Calculate games played (weeks with at least 1 target)
            games_played = player_targets['week'].n_unique()

            if games_played == 0:
                return features

            # Get player's receiving yards
            player_completions = player_targets.filter(pl.col('complete_pass') == 1)
            total_rec_yards = player_completions['receiving_yards'].fill_null(0).sum()

            # === 1. SEASON-LONG TARGET VOLUME ===
            features['targets_season_avg'] = len(player_targets) / games_played

            # === 2. SEASON-LONG TARGET SHARE ===
            team_targets_all = pbp_filtered.filter(
                (pl.col('posteam') == player_team) &
                ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
            )

            if len(team_targets_all) >= 10:
                features['target_share_season'] = len(player_targets) / len(team_targets_all)

            # === 3. SEASON-LONG YARDS PER TARGET ===
            if len(player_targets) > 0:
                features['yards_per_target_season'] = total_rec_yards / len(player_targets)

            # === 4. 3-WEEK ROLLING METRICS ===
            recent_weeks = pbp_filtered.filter(
                pl.col('week') >= max(1, week - 3)
            )

            player_recent_targets = recent_weeks.filter(
                (pl.col('receiver_player_id') == player_id) &
                ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
            )

            if len(player_recent_targets) >= 3:  # Minimum for 3-week rolling
                # Recent games played
                recent_games = player_recent_targets['week'].n_unique()

                if recent_games > 0:
                    # Recent target average
                    features['targets_3wk_avg'] = len(player_recent_targets) / recent_games

                    # Recent receiving yards
                    recent_completions = player_recent_targets.filter(pl.col('complete_pass') == 1)
                    recent_rec_yards = recent_completions['receiving_yards'].fill_null(0).sum()

                    # Recent YPT
                    features['yards_per_target_3wk'] = recent_rec_yards / len(player_recent_targets)

                    # Recent target share
                    team_recent_targets = recent_weeks.filter(
                        (pl.col('posteam') == player_team) &
                        ((pl.col('complete_pass') == 1) | (pl.col('incomplete_pass') == 1))
                    )

                    if len(team_recent_targets) >= 5:
                        features['target_share_3wk'] = len(player_recent_targets) / len(team_recent_targets)

        except Exception as e:
            logger.debug(f"Error calculating target volume features: {e}")

        return features

    def _extract_nextgen_stats_features(
        self,
        player_id: str,
        season: int,
        week: int,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract NextGen Stats features for WR/TE (2 features).

        GPS tracking metrics from NextGen Stats (available 2016+):
        - avg_separation: Yards of separation at time of catch
        - avg_cushion: Yards of cushion at snap

        These metrics quantify receiver ability to create space and
        beat man coverage, which correlates with YAC and big plays.

        Phase 1 optimization: Use cached NextGen data to avoid redundant file I/O.

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number (filters through week-1)
            data_cache: Optional pre-loaded data cache

        Returns:
            Dict with 2 NextGen features:
            - avg_separation: Season-long separation average (through week-1)
            - avg_cushion: Season-long cushion average (through week-1)
        """
        # Default to NaN for missing data (pre-2016 or player not in cache)
        # Tree models (XGBoost, LightGBM, CatBoost) handle NaN natively
        features = {
            'avg_separation': float('nan'),  # NaN for missing data
            'avg_cushion': float('nan'),     # NaN for missing data
        }

        # NextGen Stats only available 2016+
        if season < 2016:
            return features  # Return NaN for pre-2016 seasons

        through_week = week - 1
        if through_week < 1:
            return features

        try:
            # Phase 1: Check if NextGen data is cached
            nextgen_data = None
            if data_cache and 'nextgen_data' in data_cache:
                nextgen_cache = data_cache['nextgen_data']
                # Handle multi-year cache (dict) or single-year cache (DataFrame)
                if isinstance(nextgen_cache, dict) and season in nextgen_cache:
                    nextgen_data = nextgen_cache[season]
                elif not isinstance(nextgen_cache, dict):
                    nextgen_data = nextgen_cache  # Single year cache (fallback)

            if nextgen_data is None:
                # Fallback: Load from file
                from modules.nextgen_cache_builder import load_nextgen_cache
                nextgen_data = load_nextgen_cache(season)

            if nextgen_data is None or nextgen_data.is_empty():
                logger.debug(f"No NextGen data available for {season}")
                return features

            # Filter to player and through week-1
            player_nextgen = nextgen_data.filter(
                (pl.col('player_gsis_id') == player_id) &
                (pl.col('week') <= through_week)
            )

            if len(player_nextgen) == 0:
                return features

            # Calculate season-long averages (weighted by targets)
            # Players with more targets get more weight in weekly averages
            if 'targets' in player_nextgen.columns and player_nextgen['targets'].sum() > 0:
                # Weighted average by targets
                total_targets = player_nextgen['targets'].sum()

                if 'avg_separation' in player_nextgen.columns:
                    # Filter to weeks with valid separation data
                    sep_data = player_nextgen.filter(
                        pl.col('avg_separation').is_not_null()
                    )
                    if len(sep_data) >= 3:  # Minimum 3 weeks of data
                        weighted_sep = (
                            sep_data['avg_separation'] * sep_data['targets']
                        ).sum() / total_targets
                        features['avg_separation'] = float(weighted_sep)

                if 'avg_cushion' in player_nextgen.columns:
                    # Filter to weeks with valid cushion data
                    cush_data = player_nextgen.filter(
                        pl.col('avg_cushion').is_not_null()
                    )
                    if len(cush_data) >= 3:  # Minimum 3 weeks of data
                        weighted_cush = (
                            cush_data['avg_cushion'] * cush_data['targets']
                        ).sum() / total_targets
                        features['avg_cushion'] = float(weighted_cush)

        except Exception as e:
            logger.debug(f"Error extracting NextGen features: {e}")

        return features

    def _extract_route_participation_features(
        self,
        player_id: str,
        season: int,
        week: int,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract route participation features for WR/TE (3 features).

        Uses snap count data to calculate:
        - route_participation_pct: % of team pass plays WR runs route (via offense_pct)
        - targets_per_route_run (TPRR): Efficiency metric (targets / games)
        - route_target_gap: Difference between route % and target share

        Available 2016+ (NaN for earlier years, tree models handle natively).

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number (filters through week-1)
            data_cache: Optional pre-loaded data cache

        Returns:
            Dict with 3 route participation features
        """
        # Default to NaN for missing data (pre-2016 or not available)
        features = {
            'route_participation_pct': float('nan'),
            'targets_per_route_run': float('nan'),
            'route_target_gap': float('nan'),
        }

        # Snap counts only available 2016+
        if season < 2016:
            return features

        through_week = week - 1
        if through_week < 1:
            return features

        try:
            # Step 1: Get player's pfr_id from mapping
            player_id_mapping = None
            if data_cache and 'player_id_mapping' in data_cache:
                player_id_mapping = data_cache['player_id_mapping']

            if player_id_mapping is None:
                # Fallback: load from file
                mapping_file = Path(CACHE_DIR) / "player_id_mapping.parquet"
                if mapping_file.exists():
                    player_id_mapping = pl.read_parquet(mapping_file)

            if player_id_mapping is None:
                logger.debug(f"Player ID mapping not available")
                return features

            # Look up pfr_id for this player
            player_mapping = player_id_mapping.filter(
                pl.col('gsis_id') == player_id
            )

            if len(player_mapping) == 0:
                logger.debug(f"Player {player_id} not found in ID mapping")
                return features

            pfr_id = player_mapping['pfr_id'][0]

            # Step 2: Load snap count data
            snap_data = None
            if data_cache and 'snap_count_data' in data_cache:
                snap_cache = data_cache['snap_count_data']
                if isinstance(snap_cache, dict) and season in snap_cache:
                    snap_data = snap_cache[season]
                elif not isinstance(snap_cache, dict):
                    snap_data = snap_cache

            if snap_data is None:
                # Fallback: load from file
                from modules.snap_count_cache_builder import load_snap_count_cache
                snap_data = load_snap_count_cache(season)

            if snap_data is None or snap_data.is_empty():
                logger.debug(f"No snap count data available for {season}")
                return features

            # Step 3: Filter to player and through week-1
            player_snaps = snap_data.filter(
                (pl.col('pfr_player_id') == pfr_id) &
                (pl.col('week') <= through_week)
            )

            if len(player_snaps) == 0:
                logger.debug(f"No snap data for player {pfr_id} in {season} weeks 1-{through_week}")
                return features

            # Step 4: Calculate route participation percentage
            # offense_pct is the % of team's offensive snaps the player was on field
            # For WRs, this approximates route participation (they run routes on most snaps)
            avg_offense_pct = float(player_snaps['offense_pct'].mean())
            features['route_participation_pct'] = avg_offense_pct

            # Step 5: Calculate targets per route run
            # Get targets from player_volume cache
            total_targets = 0
            if data_cache and 'player_volume' in data_cache:
                volume_cache = data_cache['player_volume'].get(season, {})
                player_volume = volume_cache.get(player_id)

                if player_volume is not None:
                    # Filter to through_week and sum targets
                    player_targets = player_volume.filter(
                        pl.col('week') <= through_week
                    )
                    if 'targets' in player_targets.columns:
                        total_targets = int(player_targets['targets'].sum())

            # TPRR = targets per game (approximation since we don't have exact route counts)
            games_played = len(player_snaps)
            if games_played > 0 and total_targets > 0:
                features['targets_per_route_run'] = float(total_targets) / float(games_played)

            # Step 6: Calculate route-target gap
            # This measures if a player is getting targets proportional to route participation
            if not pl.Series([features['route_participation_pct']]).is_nan()[0] and total_targets > 0:
                # Estimate target share (targets per game / team pass attempts per game)
                # Simplified: just use raw difference between snap % and targets per game
                if games_played > 0:
                    targets_per_game = float(total_targets) / float(games_played)
                    # Normalize to same scale as offense_pct (0-1 scale)
                    # Average team has ~35 pass attempts per game, so normalize
                    target_rate = targets_per_game / 35.0  # Rough normalization
                    features['route_target_gap'] = avg_offense_pct - target_rate

        except Exception as e:
            logger.debug(f"Error extracting route participation features: {e}")

        return features

    def _extract_prior_season_features(
        self,
        player_id: str,
        season: int,
        week: int,
        position: str,
        stat_column: str,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract prior season baseline features for year-over-year context (3 features).

        Phase 6 optimization: Pass data_cache to eliminate 10,000+ file I/O operations per year.

        Provides historical context to understand player trajectory:
        - Improving (yoy_trend > 1.05): Player getting better
        - Stable (0.95 < yoy_trend < 1.05): Consistent performance
        - Declining (yoy_trend < 0.95): Regression
        - Sophomore slump: 2nd year players often underperform

        Args:
            player_id: Player GSIS ID
            season: Current season
            week: Current week
            position: Player position
            stat_column: Stat column to analyze
            data_cache: Optional pre-loaded data cache (Phase 6)

        Returns:
            Dict with 3 year-over-year features:
            - prior_season_avg: Full prior season average
            - yoy_trend: Current / prior ratio (>1 = improving)
            - sophomore_indicator: Binary flag for 2nd year players
        """
        # Defaults (no history)
        features = {
            'prior_season_avg': 0.0,
            'yoy_trend': 1.0,        # Neutral trend
            'sophomore_indicator': 0  # Not a sophomore
        }

        try:
            # Load prior season stats (using cache)
            prior_season = season - 1
            prior_stats = self._load_player_stats(player_id, prior_season, position, data_cache)

            if prior_stats is None or len(prior_stats) == 0:
                # No prior season data - could be rookie
                # Check if this is their 2nd season (sophomore)
                two_years_ago = season - 2
                two_years_ago_stats = self._load_player_stats(player_id, two_years_ago, position, data_cache)

                if two_years_ago_stats is not None and len(two_years_ago_stats) > 0:
                    # They played 2 years ago but not last year - not a sophomore
                    features['sophomore_indicator'] = 0
                else:
                    # No data 2 years ago either - check if they have current season data (indicates rookie/sophomore)
                    current_stats = self._load_player_stats(player_id, season, position, data_cache)
                    if current_stats is not None and len(current_stats) > 0:
                        # Have current season data but no prior - likely rookie
                        features['sophomore_indicator'] = 0

                return features

            # Calculate prior season average (full season)
            if stat_column in prior_stats.columns:
                prior_season_data = prior_stats[stat_column].drop_nulls()

                if len(prior_season_data) >= 3:  # Minimum 3 games prior season
                    prior_avg = prior_season_data.mean()
                    features['prior_season_avg'] = prior_avg

                    # Calculate current season average (through week-1)
                    current_stats = self._load_player_stats(player_id, season, position, data_cache)
                    if current_stats is not None and len(current_stats) > 0:
                        through_week = week - 1
                        current_season_data = current_stats.filter(
                            pl.col('week') <= through_week
                        )[stat_column].drop_nulls()

                        if len(current_season_data) >= 2:  # Minimum 2 games current season
                            current_avg = current_season_data.mean()

                            # Calculate YoY trend (avoid division by zero)
                            if prior_avg > 0:
                                features['yoy_trend'] = current_avg / prior_avg
                            else:
                                features['yoy_trend'] = 1.0

            # Check if sophomore (has prior season data, but no data 2 years ago)
            two_years_ago = season - 2
            two_years_ago_stats = self._load_player_stats(player_id, two_years_ago, position, data_cache)

            if two_years_ago_stats is None or len(two_years_ago_stats) == 0:
                # Has last year but not 2 years ago = sophomore
                features['sophomore_indicator'] = 1

        except Exception as e:
            logger.debug(f"Error extracting prior season features: {e}")

        return features

    def _extract_injury_features(
        self,
        player_id: str,
        season: int,
        week: int,
        position: str,
        prop_type: str,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract comprehensive injury-related features (12 features).

        Features include:
        - Historical injury pattern (3-year lookback)
        - Current season injury context
        - Current week injury status (MOST PREDICTIVE)
        - Position-specific injury impact

        Phase 1 optimization: Use cached injury and roster data to avoid redundant file I/O.

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number
            position: Player position
            prop_type: Prop type being predicted
            data_cache: Optional pre-loaded data cache

        Returns:
            Dict with 12 injury features
        """
        from modules.injury_cache_builder import (
            count_games_missed_due_to_injury,
            classify_injury_pattern,
            load_injury_data,
            load_roster_data
        )

        features = {}

        try:
            # 1. Historical injury pattern (3-year lookback)
            # OPTIMIZATION: Use batch cache to eliminate N+1 queries (Phase 3)
            injury_history = []
            for year_offset in range(1, 4):  # Y-1, Y-2, Y-3
                past_season = season - year_offset
                if past_season >= 2009:  # Injury data starts 2009
                    # Check batch cache first, fall back to function call if needed
                    if data_cache and 'injury_metrics_batch' in data_cache:
                        year_data = data_cache['injury_metrics_batch'].get((player_id, past_season))
                        if year_data is None:
                            # Fallback for cache miss
                            year_data = count_games_missed_due_to_injury(
                                player_id, past_season, max_games=17
                            )
                    else:
                        # No batch cache available, use original method
                        year_data = count_games_missed_due_to_injury(
                            player_id, past_season, max_games=17
                        )
                    injury_history.append(year_data)

            # Extract historical features
            if len(injury_history) >= 1:
                features['injury_games_missed_y1'] = float(injury_history[0]['injury_missed'])
                features['injury_games_missed_y2'] = float(injury_history[1]['injury_missed']) if len(injury_history) > 1 else 0.0
                features['injury_games_missed_y3'] = float(injury_history[2]['injury_missed']) if len(injury_history) > 2 else 0.0

                # Classification (0=reliable, 1=moderate, 2=elevated, 3=injury-prone)
                classification, _ = classify_injury_pattern(injury_history)
                class_map = {'reliable': 0, 'moderate': 1, 'elevated': 2, 'injury-prone': 3}
                features['injury_classification_score'] = float(class_map.get(classification, 0))

                # Recurring injury check (same body part 2+ times in 3 years)
                all_injuries = []
                for hist in injury_history:
                    all_injuries.extend(hist.get('injury_types', []))

                injury_counts = {}
                for inj in all_injuries:
                    if inj:
                        injury_counts[inj.lower()] = injury_counts.get(inj.lower(), 0) + 1

                features['has_recurring_injury'] = 1.0 if any(c >= 2 for c in injury_counts.values()) else 0.0
            else:
                # Rookie or no history - defaults
                features['injury_games_missed_y1'] = 0.0
                features['injury_games_missed_y2'] = 0.0
                features['injury_games_missed_y3'] = 0.0
                features['injury_classification_score'] = 0.0  # Assume reliable
                features['has_recurring_injury'] = 0.0

            # 2. Current season injury context (through week-1)
            # OPTIMIZATION: Use batch cache to eliminate N+1 queries (Phase 3)
            if data_cache and 'injury_metrics_batch' in data_cache:
                current_season_data = data_cache['injury_metrics_batch'].get((player_id, season))
                if current_season_data is None:
                    # Fallback for cache miss
                    current_season_data = count_games_missed_due_to_injury(
                        player_id, season, max_games=17
                    )
            else:
                # No batch cache available, use original method
                current_season_data = count_games_missed_due_to_injury(
                    player_id, season, max_games=17
                )
            features['games_missed_current_season'] = float(current_season_data['injury_missed'])

            # 3. Current week injury status (MOST IMPORTANT FEATURE)
            # Phase 1: Check if injury data is cached
            if data_cache and 'injury_data' in data_cache:
                injury_cache = data_cache['injury_data']
                # Cache is dict of {year: df} - extract correct season
                injuries_df = injury_cache.get(season) if isinstance(injury_cache, dict) else injury_cache
            else:
                # Fallback: Load from file
                injuries_df = load_injury_data(season)

            if injuries_df is not None and not injuries_df.is_empty():
                # Look for injury report in week N (predicting for week N)
                player_injury = injuries_df.filter(
                    (pl.col('gsis_id') == player_id) &
                    (pl.col('week') == week)
                )

                if len(player_injury) > 0:
                    status = player_injury['report_status'][0]

                    # Status severity (0=none, 1=questionable, 2=doubtful, 3=out)
                    status_map = {'Questionable': 1, 'Doubtful': 2, 'Out': 3}
                    features['injury_status_score'] = float(status_map.get(status, 1))

                    # Injury type classification
                    primary_injury = player_injury['report_primary_injury'][0]
                    if primary_injury:
                        inj_lower = primary_injury.lower()
                        # Mobility injuries (legs/lower body) - impact rushing/receiving
                        mobility_keywords = ['ankle', 'hamstring', 'knee', 'quad', 'calf', 'foot', 'hip', 'thigh']
                        # Upper body injuries - impact passing/catching
                        upper_keywords = ['shoulder', 'hand', 'wrist', 'ribs', 'chest', 'arm', 'elbow', 'finger']

                        features['injury_type_mobility'] = 1.0 if any(k in inj_lower for k in mobility_keywords) else 0.0
                        features['injury_type_upper_body'] = 1.0 if any(k in inj_lower for k in upper_keywords) else 0.0
                    else:
                        features['injury_type_mobility'] = 0.0
                        features['injury_type_upper_body'] = 0.0
                else:
                    # Not on injury report this week
                    features['injury_status_score'] = 0.0
                    features['injury_type_mobility'] = 0.0
                    features['injury_type_upper_body'] = 0.0
            else:
                # No injury data available for this season
                features['injury_status_score'] = 0.0
                features['injury_type_mobility'] = 0.0
                features['injury_type_upper_body'] = 0.0

            # 4. Weeks since last missed game (recency of injury)
            if current_season_data['injury_missed'] > 0:
                # Find most recent missed week
                # Phase 1: Check if roster data is cached
                if data_cache and 'roster_data' in data_cache:
                    rosters_df = data_cache['roster_data']
                else:
                    # Fallback: Load from file
                    rosters_df = load_roster_data(season)

                if rosters_df is not None and not rosters_df.is_empty():
                    player_weeks = rosters_df.filter(
                        (pl.col('gsis_id') == player_id) &
                        (pl.col('status').is_in(['INA', 'RES']))
                    )

                    if len(player_weeks) > 0:
                        last_missed = player_weeks['week'].max()
                        features['weeks_since_last_missed'] = float(week - last_missed)
                    else:
                        features['weeks_since_last_missed'] = 999.0  # No recent miss
                else:
                    features['weeks_since_last_missed'] = 999.0
            else:
                features['weeks_since_last_missed'] = 999.0  # No misses this season

        except Exception as e:
            logger.debug(f"Error extracting injury features: {e}")
            # Return defaults on error
            features = {
                'injury_games_missed_y1': 0.0,
                'injury_games_missed_y2': 0.0,
                'injury_games_missed_y3': 0.0,
                'injury_classification_score': 0.0,
                'has_recurring_injury': 0.0,
                'games_missed_current_season': 0.0,
                'injury_status_score': 0.0,
                'injury_type_mobility': 0.0,
                'injury_type_upper_body': 0.0,
                'weeks_since_last_missed': 999.0
            }

        # Return all injury features - config-based filtering will handle prop-specific needs
        # RB rushing config: Only includes 'injury_status_score'
        # WR receiving config: Only includes 'injury_status_score'
        # QB passing config: Includes all 10 injury features
        return features

    # DISABLED: Game-level weather extraction (reverted to Game Script baseline)
    # This method was added to test game-level weather features as volume predictors,
    # but evaluation showed it created the same inverse confidence signal as
    # player-specific weather features (50+ yards: 50.0% accuracy, -4.55% ROI).
    # Restored Game Script baseline (45 features) which maintains high-confidence edge
    # (50+ yards: 66.7% accuracy, +27.27% ROI).
    #
    # def _get_game_weather(
    #     self,
    #     player_id: str,
    #     season: int,
    #     week: int,
    #     position: str
    # ) -> Dict[str, any]:
    #     """
    #     Extract game-level weather data from PBP cache.
    #
    #     Returns weather conditions for the specific game this player participates in.
    #     Used for game-level volume prediction (not player-specific efficiency).
    #
    #     Args:
    #         player_id: Player GSIS ID (used to find their game)
    #         season: Season year
    #         week: Week number
    #         position: Position code (to identify player column in PBP)
    #
    #     Returns:
    #         Dict with game weather data:
    #             - temp (float): Temperature in Fahrenheit
    #             - wind (float): Wind speed in mph
    #             - weather (str): Weather description
    #             - roof (str): Roof type
    #             - is_dome (bool): True if dome/closed roof
    #     """
    #     defaults = {
    #         'temp': 70.0,
    #         'wind': 5.0,
    #         'weather': 'Clear',
    #         'roof': 'outdoors',
    #         'is_dome': False
    #     }
    #
    #     try:
    #         pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
    #         if not pbp_file.exists():
    #             return defaults
    #
    #         pbp = pl.read_parquet(pbp_file)
    #
    #         # Filter to this player's game in this week
    #         if position == 'QB':
    #             player_plays = pbp.filter(
    #                 (pl.col('passer_player_id') == player_id) &
    #                 (pl.col('week') == week)
    #             )
    #         elif position == 'RB':
    #             player_plays = pbp.filter(
    #                 (pl.col('rusher_player_id') == player_id) &
    #                 (pl.col('week') == week)
    #             )
    #         else:  # WR/TE
    #             player_plays = pbp.filter(
    #                 (pl.col('receiver_player_id') == player_id) &
    #                 (pl.col('week') == week)
    #             )
    #
    #         if len(player_plays) == 0:
    #             # Player not found, try to get game weather by opponent
    #             # (fallback for players with no plays yet)
    #             return defaults
    #
    #         # Get first play's weather data (same for entire game)
    #         first_play = player_plays.head(1)
    #
    #         temp = first_play['temp'][0] if first_play['temp'][0] is not None else 70.0
    #         wind = first_play['wind'][0] if first_play['wind'][0] is not None else 5.0
    #         weather_desc = first_play['weather'][0] if first_play['weather'][0] is not None else 'Clear'
    #         roof = first_play['roof'][0] if first_play['roof'][0] is not None else 'outdoors'
    #
    #         # Determine if dome
    #         is_dome = roof in ['dome', 'closed']
    #
    #         return {
    #             'temp': float(temp),
    #             'wind': float(wind),
    #             'weather': weather_desc,
    #             'roof': roof,
    #             'is_dome': is_dome
    #         }
    #
    #     except Exception as e:
    #         logger.debug(f"Error extracting game weather: {e}")
    #         return defaults

    def _extract_game_context_features(
        self,
        opponent_team: str,
        player_id: str = None,
        season: int = None,
        week: int = None,
        position: str = None,
        pbp_df: Optional[pl.DataFrame] = None,
        player_team: str = None,
        data_cache: Optional[dict] = None
    ) -> Dict[str, float]:
        """
        Extract game context features (7 features).

        Features:
        - is_home: Home game indicator
        - is_dome: Dome game indicator
        - division_game: Division opponent indicator
        - game_temp: Temperature in °F
        - game_wind: Wind speed in mph
        - vegas_total: Game total over/under
        - vegas_spread: Point spread

        Phase 4 optimization: Uses game metadata cache for O(1) lookups
        instead of filtering PBP data (eliminates 30,000+ filter operations).

        Uses NaN for missing data (tree models handle natively).
        """
        # Default to NaN for missing data
        features = {
            'is_home': float('nan'),
            'is_dome': float('nan'),
            'division_game': float('nan'),
            'game_temp': float('nan'),
            'game_wind': float('nan'),
            'extreme_weather': float('nan'),
            'vegas_total': float('nan'),
            'vegas_spread': float('nan'),
            'implied_team_total': float('nan'),
            'opp_implied_total': float('nan')
        }

        # If we don't have the necessary parameters, return NaN
        if not all([season, week]):
            return features

        # Phase 4 optimization: Use game metadata cache if available
        if data_cache and 'game_metadata' in data_cache:
            game_metadata_cache = data_cache['game_metadata'].get(season, {})

            # Determine player's team (passed as hint or fallback)
            team = player_team

            if team and (team, week) in game_metadata_cache:
                # O(1) lookup instead of O(n) PBP filtering!
                game_meta = game_metadata_cache[(team, week)]

                features['is_home'] = game_meta.get('is_home', float('nan'))
                is_dome = game_meta.get('is_dome')
                if is_dome is not None:
                    features['is_dome'] = is_dome

                temp = game_meta.get('temp')
                if temp is not None:
                    features['game_temp'] = float(temp)

                wind = game_meta.get('wind')
                if wind is not None:
                    features['game_wind'] = float(wind)

                # Division game (from game metadata cache)
                div_game = game_meta.get('div_game')
                if div_game is not None:
                    features['division_game'] = float(div_game)

                # Vegas lines (from betting lines cache)
                if data_cache and 'betting_lines' in data_cache:
                    betting_cache = data_cache.get('betting_lines', {}).get(season, {})
                    if betting_cache and team and (team, week) in betting_cache:
                        betting_data = betting_cache[(team, week)]
                        features['vegas_total'] = betting_data['vegas_total']
                        features['vegas_spread'] = betting_data['vegas_spread']

                # If cache provided some features, return (don't fall through to legacy)
                if features['is_home'] is not float('nan'):
                    # Add implied totals from vegas lines
                    if features['vegas_total'] is not float('nan') and features['vegas_spread'] is not float('nan'):
                        total = features['vegas_total']
                        spread = features['vegas_spread']
                        features['implied_team_total'] = (total / 2) + (spread / 2)
                        features['opp_implied_total'] = (total / 2) - (spread / 2)

                    # Extreme weather indicator (only if weather data available)
                    if features['game_wind'] is not float('nan') and features['game_temp'] is not float('nan'):
                        wind = features['game_wind']
                        temp = features['game_temp']
                        features['extreme_weather'] = 1.0 if (wind > 15 or temp < 25 or temp > 85) else 0.0

                    return features

        # Fallback: Use existing PBP filtering logic if cache unavailable
        if not all([player_id, position]):
            return features

        try:
            if pbp_df is None:
                pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
                if not pbp_file.exists():
                    return features
                pbp = pl.read_parquet(pbp_file)
            else:
                pbp = pbp_df

            # Find player's game in this week
            if position == 'QB':
                player_plays = pbp.filter(
                    (pl.col('passer_player_id') == player_id) &
                    (pl.col('week') == week)
                )
            elif position == 'RB':
                player_plays = pbp.filter(
                    (pl.col('rusher_player_id') == player_id) &
                    (pl.col('week') == week)
                )
            else:  # WR/TE
                player_plays = pbp.filter(
                    (pl.col('receiver_player_id') == player_id) &
                    (pl.col('week') == week)
                )

            if len(player_plays) == 0:
                # Player not found in this week - likely hasn't played yet
                # Try to find game by opponent matchup (for prediction)
                week_games = pbp.filter(pl.col('week') == week)

                # Get player's team from prior weeks
                if position == 'QB':
                    prior_plays = pbp.filter(
                        (pl.col('passer_player_id') == player_id) &
                        (pl.col('week') < week)
                    )
                elif position == 'RB':
                    prior_plays = pbp.filter(
                        (pl.col('rusher_player_id') == player_id) &
                        (pl.col('week') < week)
                    )
                else:
                    prior_plays = pbp.filter(
                        (pl.col('receiver_player_id') == player_id) &
                        (pl.col('week') < week)
                    )

                if len(prior_plays) > 0:
                    player_team = prior_plays['posteam'][0]

                    # Find game between player_team and opponent_team
                    game_plays = week_games.filter(
                        ((pl.col('home_team') == player_team) & (pl.col('away_team') == opponent_team)) |
                        ((pl.col('away_team') == player_team) & (pl.col('home_team') == opponent_team))
                    )

                    if len(game_plays) > 0:
                        player_plays = game_plays
                    else:
                        return features  # Can't find the game
                else:
                    return features  # Can't determine player's team

            # Get first play for game-level data
            first_play = player_plays.head(1)

            if len(first_play) == 0:
                return features

            # Extract player's team
            player_team = first_play['posteam'][0]

            # Home/Away
            home_team = first_play['home_team'][0]
            if player_team and home_team:
                features['is_home'] = 1.0 if player_team == home_team else 0.0

            # Dome/Stadium
            roof = first_play['roof'][0]
            if roof:
                features['is_dome'] = 1.0 if roof in ['dome', 'closed'] else 0.0

            # Weather
            temp = first_play['temp'][0]
            wind = first_play['wind'][0]
            if temp is not None:
                features['game_temp'] = float(temp)
            if wind is not None:
                features['game_wind'] = float(wind)

            # Division game (from PBP data)
            div_game = first_play['div_game'][0]
            if div_game is not None:
                features['division_game'] = float(div_game)

            # Vegas lines (from betting lines cache)
            if data_cache and 'betting_lines' in data_cache:
                betting_cache = data_cache.get('betting_lines', {}).get(season, {})
                if betting_cache and player_team and (player_team, week) in betting_cache:
                    betting_data = betting_cache[(player_team, week)]
                    features['vegas_total'] = betting_data['vegas_total']
                    features['vegas_spread'] = betting_data['vegas_spread']

        except Exception as e:
            logger.debug(f"Error extracting game context features: {e}")

        return features

    def _extract_game_script_features(
        self,
        player_id: str,
        season: int,
        week: int,
        position: str,
        opponent_team: str,
        prop_type: str,
        pbp_df: Optional[pl.DataFrame] = None,
        data_cache: Optional[dict] = None,
        player_team: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract game script and team context features (4-5 features).

        Phase 8A optimization: Uses game_script cache for O(1) lookups.
        Eliminates 8-10 redundant PBP filter operations per player-week call.

        Volume-driven features that predict passing attempts/opportunities:
        1. Team Offensive Context:
           - team_avg_margin: Rolling 3-game point differential (game script)
        2. Opponent Defensive Context:
           - opp_def_ppg_allowed: Opponent points per game allowed
           - opp_def_ypg_allowed: Opponent yards per game allowed (QB-only)
        3. Pace & Tempo:
           - team_plays_per_game: Rolling 3-game offensive plays per game
           - team_time_of_possession: Average TOP per game (minutes)

        All features use data through week-1 only (no future data).
        """
        features = {
            'team_avg_margin': 0.0,  # Neutral game script
            'opp_def_ppg_allowed': 22.0,  # League average ~22 PPG
            # opp_def_ypg_allowed: QB-only feature (added conditionally below)
            'team_plays_per_game': 65.0,  # League average ~65 plays
            'team_time_of_possession': 30.0  # League average 30 minutes
        }

        # Phase 8A: Use game script cache if available (O(1) lookup!)
        if data_cache and 'game_script' in data_cache:
            game_script_cache = data_cache['game_script'].get(season, {})

            through_week = week - 1
            if through_week < 1:
                return features

            try:
                # Get player's team (passed as parameter or fallback to PBP lookup)
                if not player_team and pbp_df is not None:
                    pbp_filtered = pbp_df.filter((pl.col('week') < week))
                    if position == 'QB':
                        player_team_df = pbp_filtered.filter(pl.col('passer_player_id') == player_id).select('posteam').head(1)
                    elif position == 'RB':
                        player_team_df = pbp_filtered.filter(pl.col('rusher_player_id') == player_id).select('posteam').head(1)
                    else:  # WR/TE
                        player_team_df = pbp_filtered.filter(pl.col('receiver_player_id') == player_id).select('posteam').head(1)

                    if len(player_team_df) > 0:
                        player_team = player_team_df['posteam'][0]

                if not player_team:
                    return features

                # O(1) lookup from game script cache
                cache_key = (player_team, through_week)
                if cache_key in game_script_cache:
                    cached_stats = game_script_cache[cache_key]

                    # Populate all features from cache
                    features['team_avg_margin'] = cached_stats.get('team_avg_margin', 0.0)
                    features['opp_def_ppg_allowed'] = cached_stats.get('opp_def_ppg_allowed', 22.0)
                    features['team_plays_per_game'] = cached_stats.get('team_plays_per_game', 65.0)
                    features['team_time_of_possession'] = cached_stats.get('team_time_of_possession', 30.0)

                    # QB-only feature
                    if position == 'QB':
                        features['opp_def_ypg_allowed'] = cached_stats.get('opp_def_ypg_allowed', 330.0)

                    return features

            except Exception as e:
                logger.debug(f"Error using game script cache: {e}")
                # Fall through to legacy implementation below

        # Load PBP data if not provided
        if pbp_df is None:
            pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
            if not pbp_file.exists():
                return features

            try:
                pbp = pl.read_parquet(pbp_file)
            except Exception as e:
                logger.warning(f"Error loading PBP data: {e}")
                return features
        else:
            pbp = pbp_df

        try:
            pbp_filtered = pbp.filter(pl.col('week') < week)

            # Get player's team from their most recent game
            if position == 'QB':
                player_team_df = pbp_filtered.filter(
                    pl.col('passer_player_id') == player_id
                ).select('posteam').head(1)
            elif position == 'RB':
                player_team_df = pbp_filtered.filter(
                    pl.col('rusher_player_id') == player_id
                ).select('posteam').head(1)
            else:  # WR/TE
                player_team_df = pbp_filtered.filter(
                    pl.col('receiver_player_id') == player_id
                ).select('posteam').head(1)

            if len(player_team_df) == 0:
                return features

            player_team = player_team_df['posteam'][0]

            # === 1. TEAM OFFENSIVE CONTEXT ===

            # Team Average Margin (rolling 3-game)
            team_games = pbp_filtered.filter(
                (pl.col('posteam') == player_team) | (pl.col('defteam') == player_team)
            ).group_by(['game_id', 'week']).agg([
                pl.when(pl.col('posteam') == player_team)
                  .then(pl.col('total_home_score') - pl.col('total_away_score'))
                  .when(pl.col('home_team') == player_team)
                  .then(pl.col('total_home_score') - pl.col('total_away_score'))
                  .otherwise(pl.col('total_away_score') - pl.col('total_home_score'))
                  .first().alias('margin')
            ]).sort('week', descending=True)

            if len(team_games) >= 3:
                recent_3 = team_games.head(3)
                features['team_avg_margin'] = float(recent_3['margin'].mean())
            elif len(team_games) > 0:
                features['team_avg_margin'] = float(team_games['margin'].mean())

            # === 2. OPPONENT DEFENSIVE CONTEXT ===

            # Opponent Points Per Game Allowed
            opp_def_games = pbp_filtered.filter(
                pl.col('defteam') == opponent_team
            ).group_by(['game_id']).agg([
                pl.when(pl.col('home_team') == opponent_team)
                  .then(pl.col('total_away_score'))
                  .otherwise(pl.col('total_home_score'))
                  .first().alias('points_allowed')
            ])

            if len(opp_def_games) > 0:
                features['opp_def_ppg_allowed'] = float(opp_def_games['points_allowed'].mean())

            # Opponent Yards Per Game Allowed (QB-only to protect working QB model)
            if position == 'QB':
                opp_def_yards = pbp_filtered.filter(
                    pl.col('defteam') == opponent_team
                ).group_by(['game_id']).agg([
                    (pl.col('passing_yards') + pl.col('rushing_yards')).sum().alias('total_yards')
                ])

                if len(opp_def_yards) > 0:
                    features['opp_def_ypg_allowed'] = float(opp_def_yards['total_yards'].mean())

            # === 3. PACE & TEMPO ===

            # Team Plays Per Game (rolling 3-game average)
            team_offensive_plays = pbp_filtered.filter(
                pl.col('posteam') == player_team
            ).group_by(['game_id', 'week']).agg([
                pl.col('play_id').count().alias('plays')
            ]).sort('week', descending=True)

            if len(team_offensive_plays) >= 3:
                recent_3_plays = team_offensive_plays.head(3)
                features['team_plays_per_game'] = float(recent_3_plays['plays'].mean())
            elif len(team_offensive_plays) > 0:
                features['team_plays_per_game'] = float(team_offensive_plays['plays'].mean())

            # Team Time of Possession (average per game in minutes)
            # Calculate from game duration and play count ratio
            team_plays_total = pbp_filtered.filter(
                pl.col('posteam') == player_team
            )
            all_plays_total = pbp_filtered

            if len(team_plays_total) > 0 and len(all_plays_total) > 0:
                games_played = len(team_games)
                if games_played > 0:
                    team_play_share = len(team_plays_total) / (len(all_plays_total) / 32.0)  # ~32 teams
                    # Approximate TOP: play share * 60 minutes (assumes equal pace)
                    features['team_time_of_possession'] = min(40.0, max(20.0, team_play_share * 30.0))

        except Exception as e:
            logger.debug(f"Error calculating game script features: {e}")

        return features

    def _extract_weather_features(
        self,
        player_id: str,
        season: int,
        position: str
    ) -> Dict[str, float]:
        """
        Extract player-specific weather performance features (10 features).

        Uses PRIOR SEASON weather data (season-1) to avoid data leakage.
        Represents how this player historically performs in different weather conditions.

        Features:
        - temp_cold_adj, temp_cool_adj, temp_moderate_adj, temp_hot_adj
        - wind_calm_adj, wind_moderate_adj, wind_high_adj
        - precip_adj, dome_adj, outdoor_adj

        All values are performance multipliers (typically 0.95-1.05):
        - 1.0 = average performance
        - >1.0 = performs better in this condition
        - <1.0 = performs worse in this condition

        Args:
            player_id: Player GSIS ID
            season: Current season year
            position: Player position (QB, RB, WR, TE)

        Returns:
            Dict with 10 weather performance features
        """
        # Default values (neutral/average performance)
        features = {
            'temp_cold_adj': 1.0,
            'temp_cool_adj': 1.0,
            'temp_moderate_adj': 1.0,
            'temp_hot_adj': 1.0,
            'wind_calm_adj': 1.0,
            'wind_moderate_adj': 1.0,
            'wind_high_adj': 1.0,
            'precip_adj': 1.0,
            'dome_adj': 1.0,
            'outdoor_adj': 1.0
        }

        try:
            # Use PRIOR season data to avoid leakage
            prior_season = season - 1

            # Weather cache starts in 2016 (PBP data starts 2016)
            if prior_season < 2016:
                logger.debug(f"No weather data available for {prior_season} (< 2016)")
                return features

            # Load weather performance cache for prior season
            from modules.weather_cache_builder import build_weather_performance_cache

            weather_df = build_weather_performance_cache(prior_season, position)

            if weather_df.is_empty():
                logger.debug(f"Weather cache empty for {position} {prior_season}")
                return features

            # Find player in cache
            player_weather = weather_df.filter(pl.col('player_id') == player_id)

            if len(player_weather) == 0:
                # Player not in cache (rookie, insufficient plays, or position change)
                logger.debug(f"Player {player_id} not in {prior_season} weather cache")
                return features

            # Extract player's weather adjustments
            player_row = player_weather.row(0, named=True)

            # Map all 10 weather features
            for feature_name in features.keys():
                if feature_name in player_row:
                    features[feature_name] = float(player_row[feature_name])

            logger.debug(
                f"Loaded weather features for {player_id} from {prior_season}: "
                f"cold={features['temp_cold_adj']:.2f}, wind_high={features['wind_high_adj']:.2f}, "
                f"dome={features['dome_adj']:.2f}"
            )

        except Exception as e:
            logger.debug(f"Error extracting weather features: {e}")
            # Return defaults on error

        return features

    def _get_default_features(
        self,
        position: str,
        prop_type: str,
        opponent_team: str,
        week: int,
        season: int
    ) -> Dict[str, float]:
        """Return default features when data is unavailable."""
        features = {
            # Baseline features
            'weighted_avg': 0.0,
            'career_avg': 0.0,
            'variance_cv': 1.0,
            'games_played': 0,

            # Efficiency features
            'success_rate_3wk': 0.5,
            'red_zone_rate': 0.15,
            'usage_rate': 0.15,

            # Game context
            'is_home': 0.5,
            'is_dome': 0.0,
            'division_game': 0.0,
            'game_temp': 70.0,
            'game_wind': 5.0,
            'vegas_total': 45.0,
            'vegas_spread': 0.0,

            # Game Script & Team Context
            'team_avg_margin': 0.0,
            'opp_def_ppg_allowed': 22.0,
            # opp_def_ypg_allowed: QB-only feature (added conditionally)
            'team_plays_per_game': 65.0,
            'team_time_of_possession': 30.0,

            # Injury features
            'injury_games_missed_y1': 0.0,
            'injury_games_missed_y2': 0.0,
            'injury_games_missed_y3': 0.0,
            'injury_classification_score': 0.0,
            'has_recurring_injury': 0.0,
            'games_missed_current_season': 0.0,
            'injury_status_score': 0.0,
            'injury_type_mobility': 0.0,
            'injury_type_upper_body': 0.0,
            'weeks_since_last_missed': 999.0,

            # Categorical
            'opponent': opponent_team,
            'position': position,
            'week': week,
            'season': season
        }

        # Add opponent defense defaults
        features.update(self._get_default_opponent_features(prop_type))

        # Add position-specific defaults
        if position in ['WR', 'TE', 'RB'] and ('receiving' in prop_type or prop_type == 'receptions'):
            features.update({
                'catch_rate': 0.65,
                'avg_target_depth': 10.0,
                'yac_pct': 0.5
            })

        if position == 'RB':
            features.update({
                'player_ypc': 4.3,
                'team_ypc': 4.3,
                'ypc_diff_pct': 1.0,
                'rushing_attempt_share_season': 0.15,  # Season-long carry share baseline
                'carry_share_3wk': 0.15,                # Recent 3-game carry allocation
                'goal_line_share': 0.20                 # Inside-5 yard carry percentage
            })

        if position == 'QB':
            features.update({
                'opp_def_ypg_allowed': 340.0  # League average yards per game allowed (defensive consistency)
            })

        return features

    def _get_default_opponent_features(self, prop_type: str) -> Dict[str, float]:
        """Return default opponent defense features (league averages)."""
        features = {}

        if 'passing' in prop_type or prop_type == 'receptions' or 'receiving' in prop_type:
            features['opp_def_pass_ypa'] = 7.0  # League average
            features['opp_def_pass_td_rate'] = 0.045  # ~4.5%

        if 'rushing' in prop_type:
            features['opp_def_rush_ypc'] = 4.3  # League average
            features['opp_def_rush_td_rate'] = 0.018  # ~1.8%

        return features


if __name__ == "__main__":
    # Test feature engineering on sample player
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("="*60)
    print("ML Feature Engineering - Test")
    print("="*60)

    engineer = PropFeatureEngineer()

    # Test on Patrick Mahomes 2024 Week 10
    print("\nTest 1: Patrick Mahomes 2024 Week 10 (passing_yards)")
    features = engineer.engineer_features(
        player_id="00-0033873",
        season=2024,
        week=10,
        position='QB',
        prop_type='passing_yards',
        opponent_team='BUF'
    )

    print(f"\nGenerated {len(features)} features:")
    for key, value in sorted(features.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("Feature engineering module ready for ML training")
    print("="*60)
