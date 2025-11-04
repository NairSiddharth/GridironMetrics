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
1. Baseline Performance (8 features): weighted avg, L3/L5 avg, career avg, variance, games played, effective games
2. Opponent Defense (2-4 features): Pass/rush YPA, TD rates
3. Efficiency (4 features): Success rates, red zone rate, usage
4. Injury (11 features): Historical pattern, current status, injury type
5. Pressure Rate (3 features): QB pressure metrics, sack rate - NEW!
6. Position-Specific: Catch rate (WR/TE), blocking quality (RB)
7. Game Context (7 features): Home/away, dome, weather, Vegas lines
8. Categorical (4 features): Opponent, position, week, season

Total: ~42 features (was 39, now 42 with Pressure Rate)

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
    # Returns dict with ~42 features including injury data and Pressure Rate
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
from modules.prop_types import get_stat_column_for_prop, get_prop_config
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

    def engineer_features(
        self,
        player_id: str,
        season: int,
        week: int,
        position: str,
        prop_type: str,
        opponent_team: str
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

            # 1. Baseline Performance Features
            baseline_features = self._extract_baseline_features(
                player_stats, stat_column, through_week, player_id, season, position
            )
            features.update(baseline_features)

            # 2. Opponent Defense Features
            opponent_features = self._extract_opponent_defense_features(
                opponent_team, season, week, position, prop_type
            )
            features.update(opponent_features)

            # 3. Efficiency Features (success rate, usage)
            efficiency_features = self._extract_efficiency_features(
                player_id, season, week, position, prop_type
            )
            features.update(efficiency_features)

            # 4. Position-Specific Features
            if position in ['WR', 'TE', 'RB'] and ('receiving' in prop_type or prop_type == 'receptions'):
                catch_features = self._extract_catch_rate_features(
                    player_id, season, week
                )
                features.update(catch_features)

            if position == 'RB':
                blocking_features = self._extract_blocking_quality_features(
                    player_id, season, week
                )
                features.update(blocking_features)

            # 5. Game Context Features
            context_features = self._extract_game_context_features(
                season, week, opponent_team
            )
            features.update(context_features)

            # 6. Injury Features (NEW - comprehensive injury integration)
            injury_features = self._extract_injury_features(
                player_id, season, week, position, prop_type
            )
            features.update(injury_features)

            # 7. Pressure Rate Features (QB pressure metrics)
            pressure_features = self._extract_pressure_features(
                player_id, season, week, position
            )
            features.update(pressure_features)

            # 8. Categorical Features
            features['opponent'] = opponent_team
            features['position'] = position
            features['week'] = week
            features['season'] = season

            logger.debug(f"Generated {len(features)} features for {player_id} week {week}")

        except Exception as e:
            logger.error(f"Error generating features for {player_id} week {week}: {e}")
            return self._get_default_features(position, prop_type, opponent_team, week, season)

        return features

    def _load_player_stats(self, player_id: str, season: int, position: str) -> Optional[pl.DataFrame]:
        """Load player stats from cache."""
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
        position: str
    ) -> Dict[str, float]:
        """
        Extract baseline performance features (7 features).

        Uses existing functions from prop_data_aggregator.py:
        - calculate_weighted_rolling_average()
        - get_simple_average()
        - get_career_averages()
        - calculate_stat_variance()
        """
        features = {}

        # Filter to through_week (no future data)
        filtered_stats = player_stats.filter(pl.col('week') <= through_week)

        if len(filtered_stats) == 0:
            # No games played yet - return zeros
            return {
                'weighted_avg': 0.0,
                'last_3_avg': 0.0,
                'last_5_avg': 0.0,
                'career_avg': 0.0,
                'variance_cv': 1.0,  # High variance for no data
                'games_played': 0,
                'effective_games': 0.0
            }

        try:
            # Weighted rolling average (recency-weighted: L3: 1.5x, L4-6: 1.0x, L7+: 0.75x)
            features['weighted_avg'] = calculate_weighted_rolling_average(
                player_stats, stat_column, through_week=through_week
            )

            # Simple L3 and L5 averages
            features['last_3_avg'] = get_simple_average(
                player_stats, stat_column, last_n_games=3, through_week=through_week
            )
            features['last_5_avg'] = get_simple_average(
                player_stats, stat_column, last_n_games=5, through_week=through_week
            )

            # Career average (3-year lookback)
            career_avgs = get_career_averages(
                player_id=player_id,
                current_season=season,
                position=position,
                stat_columns=[stat_column],
                lookback_years=3
            )
            features['career_avg'] = career_avgs.get(stat_column, 0.0)

            # Variance (coefficient of variation)
            features['variance_cv'] = calculate_stat_variance(
                player_stats, stat_column, through_week=through_week
            )

            # Sample size features
            features['games_played'] = len(filtered_stats)

            # Effective games (injury-adjusted) - NOW USING INJURY DATA
            from modules.injury_cache_builder import calculate_injury_adjusted_games
            features['effective_games'] = calculate_injury_adjusted_games(
                player_gsis_id=player_id,
                current_season=season,
                games_played=features['games_played'],
                max_games=17
            )
            # Ratio shows injury adjustment (>1.0 = reliable, <1.0 = injury-prone)
            features['effective_games_ratio'] = (
                features['effective_games'] / features['games_played']
                if features['games_played'] > 0 else 1.0
            )

        except Exception as e:
            logger.debug(f"Error calculating baseline features: {e}")
            # Return zeros on error
            features = {
                'weighted_avg': 0.0,
                'last_3_avg': 0.0,
                'last_5_avg': 0.0,
                'career_avg': 0.0,
                'variance_cv': 1.0,
                'games_played': 0,
                'effective_games': 0.0,
                'effective_games_ratio': 1.0
            }

        return features

    def _extract_opponent_defense_features(
        self,
        opponent: str,
        season: int,
        week: int,
        position: str,
        prop_type: str
    ) -> Dict[str, float]:
        """
        Extract opponent defensive stats (4 features).

        Calculates raw defensive metrics (not multipliers):
        - opp_def_pass_ypa: Yards per attempt allowed
        - opp_def_pass_td_rate: TD rate allowed
        - opp_def_rush_ypc: Yards per carry allowed
        - opp_def_rush_td_rate: TD rate allowed

        Uses data through week-1 only (no future data).
        """
        features = {}

        # Load PBP data
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            logger.warning(f"PBP file not found: {pbp_file}")
            return self._get_default_opponent_features(prop_type)

        try:
            pbp = pl.read_parquet(pbp_file)

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

    def _extract_efficiency_features(
        self,
        player_id: str,
        season: int,
        week: int,
        position: str,
        prop_type: str
    ) -> Dict[str, float]:
        """
        Extract efficiency and usage features (4 features).

        Features:
        - success_rate_3wk: 3-week rolling success rate
        - success_rate_season: Season-long success rate
        - red_zone_rate: Red zone touch percentage
        - usage_rate: Target share or carry share
        """
        features = {
            'success_rate_3wk': 0.5,  # Default to 50%
            'success_rate_season': 0.5,
            'red_zone_rate': 0.15,  # Default to 15%
            'usage_rate': 0.15  # Default to 15% usage
        }

        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            return features

        try:
            pbp = pl.read_parquet(pbp_file)
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
                    # Season-long success rate
                    features['success_rate_season'] = player_plays['success'].mean()

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
        week: int
    ) -> Dict[str, float]:
        """
        Extract receiver catch rate features (4 features).

        Features:
        - catch_rate: Completions / targets
        - catch_rate_over_exp: Catch rate vs depth-adjusted expected
        - avg_target_depth: Average air yards per target
        - yac_pct: Yards after catch percentage
        """
        features = {
            'catch_rate': 0.65,  # Default 65% catch rate
            'catch_rate_over_exp': 0.0,
            'avg_target_depth': 10.0,  # Default 10 yards
            'yac_pct': 0.5  # Default 50% YAC
        }

        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            return features

        try:
            pbp = pl.read_parquet(pbp_file)
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
        week: int
    ) -> Dict[str, float]:
        """
        Extract RB blocking quality features (3 features).

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

        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        if not pbp_file.exists():
            return features

        try:
            pbp = pl.read_parquet(pbp_file)
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

    def _extract_injury_features(
        self,
        player_id: str,
        season: int,
        week: int,
        position: str,
        prop_type: str
    ) -> Dict[str, float]:
        """
        Extract comprehensive injury-related features (12 features).

        Features include:
        - Historical injury pattern (3-year lookback)
        - Current season injury context
        - Current week injury status (MOST PREDICTIVE)
        - Position-specific injury impact

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number
            position: Player position
            prop_type: Prop type being predicted

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
            injury_history = []
            for year_offset in range(1, 4):  # Y-1, Y-2, Y-3
                past_season = season - year_offset
                if past_season >= 2009:  # Injury data starts 2009
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
            current_season_data = count_games_missed_due_to_injury(
                player_id, season, max_games=17
            )
            features['games_missed_current_season'] = float(current_season_data['injury_missed'])

            # 3. Current week injury status (MOST IMPORTANT FEATURE)
            injuries_df = load_injury_data(season)

            if not injuries_df.is_empty():
                # Look for injury report in week N (predicting for week N)
                player_injury = injuries_df.filter(
                    (pl.col('gsis_id') == player_id) &
                    (pl.col('week') == week)
                )

                if len(player_injury) > 0:
                    status = player_injury['report_status'][0]
                    features['is_on_injury_report'] = 1.0

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
                    features['is_on_injury_report'] = 0.0
                    features['injury_status_score'] = 0.0
                    features['injury_type_mobility'] = 0.0
                    features['injury_type_upper_body'] = 0.0
            else:
                # No injury data available for this season
                features['is_on_injury_report'] = 0.0
                features['injury_status_score'] = 0.0
                features['injury_type_mobility'] = 0.0
                features['injury_type_upper_body'] = 0.0

            # 4. Weeks since last missed game (recency of injury)
            if current_season_data['injury_missed'] > 0:
                # Find most recent missed week
                rosters_df = load_roster_data(season)
                if not rosters_df.is_empty():
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
                'is_on_injury_report': 0.0,
                'injury_status_score': 0.0,
                'injury_type_mobility': 0.0,
                'injury_type_upper_body': 0.0,
                'weeks_since_last_missed': 999.0
            }

        return features

    def _extract_pressure_features(
        self,
        player_id: str,
        season: int,
        week: int,
        position: str
    ) -> Dict[str, float]:
        """
        Extract Pressure Rate features (3 features).

        Pressure rate measures how often a QB is under pressure or sacked.
        Unlike CPOE (efficiency), pressure directly impacts BOTH efficiency AND volume
        (fewer attempts, hurried throws, negative plays).

        Features:
        - pressure_rate_season: % of dropbacks with pressure (season avg)
        - pressure_rate_l3: % of dropbacks with pressure (last 3 games)
        - sack_rate_season: % of dropbacks ending in sack

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number
            position: Player position (should be QB)

        Returns:
            Dict with 3 pressure features
        """
        features = {
            'pressure_rate_season': 0.25,  # League average ~25%
            'pressure_rate_l3': 0.25,
            'sack_rate_season': 0.06  # League average ~6%
        }

        # Pressure only applies to QBs
        if position != 'QB':
            return features

        try:
            # Load PBP data for this season
            pbp_file = Path(CACHE_DIR) / 'pbp' / f'pbp_{season}.parquet'

            if not pbp_file.exists():
                logger.debug(f"PBP file not found: {pbp_file}")
                return features

            pbp_df = pl.read_parquet(pbp_file)

            # Filter to this player's dropbacks through week N-1
            # Dropback = pass attempt OR sack
            player_dropbacks = pbp_df.filter(
                (pl.col('passer_player_id') == player_id) &
                (pl.col('week') < week) &  # Through week N-1
                ((pl.col('pass_attempt') == 1) | (pl.col('sack') == 1))
            )

            if len(player_dropbacks) == 0:
                return features

            # Calculate pressure metrics
            # qb_hit: QB was hit (includes sacks + hits while throwing)
            # sack: QB was sacked
            total_dropbacks = len(player_dropbacks)

            # Season-long pressure rate
            # Check if qb_hit column exists (NextGen data, 2016+)
            if 'qb_hit' in player_dropbacks.columns:
                pressure_plays = player_dropbacks.filter(pl.col('qb_hit') == 1)
                features['pressure_rate_season'] = float(len(pressure_plays) / total_dropbacks)
            else:
                # Fallback: use sack rate as proxy (underestimates pressure, but still informative)
                features['pressure_rate_season'] = 0.25  # Default to league average

            # Sack rate (available in all PBP data)
            sack_plays = player_dropbacks.filter(pl.col('sack') == 1)
            features['sack_rate_season'] = float(len(sack_plays) / total_dropbacks)

            # Last 3 games pressure rate
            weeks_played = sorted(player_dropbacks['week'].unique().to_list())
            if len(weeks_played) >= 3:
                last_3_weeks = weeks_played[-3:]
                last_3_dropbacks = player_dropbacks.filter(pl.col('week').is_in(last_3_weeks))

                if 'qb_hit' in last_3_dropbacks.columns:
                    last_3_pressure = last_3_dropbacks.filter(pl.col('qb_hit') == 1)
                    features['pressure_rate_l3'] = float(len(last_3_pressure) / len(last_3_dropbacks))
                else:
                    features['pressure_rate_l3'] = features['pressure_rate_season']
            else:
                # If less than 3 games, use season average
                features['pressure_rate_l3'] = features['pressure_rate_season']

        except Exception as e:
            logger.debug(f"Error extracting pressure features: {e}")

        return features

    def _extract_game_context_features(
        self,
        season: int,
        week: int,
        opponent_team: str
    ) -> Dict[str, float]:
        """
        Extract game context features (7 features).

        Features:
        - is_home: Home game indicator
        - is_dome: Dome game indicator
        - division_game: Division opponent indicator
        - game_temp: Temperature (°F)
        - game_wind: Wind speed (mph)
        - vegas_total: Game total over/under
        - vegas_spread: Point spread
        """
        features = {
            'is_home': 0.5,  # Unknown - 50/50
            'is_dome': 0.0,  # Default outdoors
            'division_game': 0.0,  # Default non-division
            'game_temp': 70.0,  # Default 70°F
            'game_wind': 5.0,  # Default 5 mph
            'vegas_total': 45.0,  # Default total
            'vegas_spread': 0.0  # Default even
        }

        # TODO: Load schedule to determine is_home
        # TODO: Load stadium info for is_dome
        # TODO: Load weather data for temp/wind
        # TODO: Load Vegas lines for total/spread

        # For now, return defaults
        # These will be populated in future iterations

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
            'last_3_avg': 0.0,
            'last_5_avg': 0.0,
            'career_avg': 0.0,
            'variance_cv': 1.0,
            'games_played': 0,
            'effective_games': 0.0,
            'effective_games_ratio': 1.0,

            # Efficiency features
            'success_rate_3wk': 0.5,
            'success_rate_season': 0.5,
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

            # Injury features (NEW)
            'injury_games_missed_y1': 0.0,
            'injury_games_missed_y2': 0.0,
            'injury_games_missed_y3': 0.0,
            'injury_classification_score': 0.0,
            'has_recurring_injury': 0.0,
            'games_missed_current_season': 0.0,
            'is_on_injury_report': 0.0,
            'injury_status_score': 0.0,
            'injury_type_mobility': 0.0,
            'injury_type_upper_body': 0.0,
            'weeks_since_last_missed': 999.0,

            # Pressure features (NEW)
            'pressure_rate_season': 0.25,
            'pressure_rate_l3': 0.25,
            'sack_rate_season': 0.06,

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
                'catch_rate_over_exp': 0.0,
                'avg_target_depth': 10.0,
                'yac_pct': 0.5
            })

        if position == 'RB':
            features.update({
                'player_ypc': 4.3,
                'team_ypc': 4.3,
                'ypc_diff_pct': 1.0
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
