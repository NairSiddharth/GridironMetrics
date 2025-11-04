# ML-Based Player Props System - Implementation Plan

**Date:** November 3, 2025
**Status:** Approved for Implementation
**Approach:** Replace rule-based projection system with machine learning ensemble

---

## Executive Summary

Replace the current rule-based player props system (37-38% win rate, -33% to -35% ROI) with a machine learning ensemble approach. The system will train on 2015-2023 player performance data using features extracted from existing calculation functions, then predict actual stat outcomes to compare against betting lines.

**Key Strategy:** Maximize reuse of existing feature calculation code (~1,000 lines) while replacing only the projection logic with ML models.

**Target Performance:** >55% win rate (vs 52.4% break-even), +5% ROI minimum

---

## Codebase Examination Results

### Data Availability

| Data Type | Years Available | Volume | Primary Use |
|-----------|----------------|---------|-------------|
| **Positional Player Stats** | 2000-2025 (26 years) | 50K+ player-weeks | PRIMARY TRAINING DATA |
| **Betting Lines** | 2023-2025 (3 years) | 3K+ props | Evaluation only (limited) |
| **Play-by-Play** | 2000-2025 (26 years) | Millions of plays | Feature engineering |
| **Weather** | 2000-2024 (25 years) | 20K+ games | Contextual features |
| **Schedules** | 2024 only | 272 games | Need 2015-2023 generation |

**Key Insight:** Train on actual stat outcomes (abundant 2015-2023 data), not betting lines (only 2023-2024 available).

### Reusable Code Identified

**HIGH VALUE REUSE** - Extract features from existing functions:

#### From `modules/prop_data_aggregator.py` (ALL functions reusable)
- `calculate_weighted_rolling_average()` → Feature: `weighted_avg_{stat}`
- `get_simple_average()` → Features: `last_3_avg`, `last_5_avg`
- `get_career_averages()` → Feature: `career_avg_{stat}`
- `calculate_stat_variance()` → Feature: `variance_cv_{stat}`

#### From `modules/prop_projection_engine.py` (extract raw stats, not multipliers)
- Opponent defense calculations → Features: `opp_def_pass_ypa`, `opp_def_rush_ypc`
- Success rate calculations → Feature: `success_rate_3wk`
- Catch rate calculations → Features: `catch_rate`, `catch_rate_over_exp`
- Route location → Feature: `red_zone_rate`

#### From `modules/adjustment_pipeline.py`
- `calculate_success_rate_adjustments_batch()` → Extract raw success rates
- `calculate_route_location_adjustments()` → Route distribution features
- `calculate_turnover_attribution_penalties_batch()` → Turnover tracking

#### From `modules/context_adjustments.py`
- `calculate_catch_rate_adjustment()` → Extract `actual_catch_rate`, `avg_depth`
- `calculate_blocking_quality_proxy()` → Extract `player_ypc`, `team_ypc`, `ypc_diff_pct`
- `calculate_yac_multiplier()` → Extract `yac_pct`

**REPLACE** - Rule-based projection logic:
- `prop_projection_engine.generate_projection()` - Replace with ML model `predict()`
- Baseline 80/20 blend formula - ML learns optimal weighting
- Adjustment multipliers (0.85x - 1.15x ranges) - ML learns feature combinations

**KEEP** - Infrastructure (no changes):
- `modules/prop_evaluator.py` - Bet evaluation (minor 1-line modification)
- `modules/prop_types.py` - Prop type configuration
- `modules/prop_output_formatter.py` - Output formatting
- `backtest_props.py` - Backtesting infrastructure (minor 1-line modification)

---

## Model Architecture

### Prop Type Models

**7 separate models** (one per prop type, position as categorical feature):

1. **passing_yards** - QB only
2. **passing_tds** - QB only
3. **rushing_yards** - QB + RB
4. **rushing_tds** - QB + RB
5. **receptions** - RB + WR + TE
6. **receiving_yards** - RB + WR + TE
7. **receiving_tds** - RB + WR + TE

**Rationale:** Prop-specific models (vs 15 position-prop combos) to maximize training data per model.

### 4-Model Ensemble per Prop Type

Each of 7 models is an ensemble of:

| Model | Purpose | Strengths |
|-------|---------|-----------|
| **XGBoost** | Gradient boosting (depth-wise) | Fast, regularized, handles missing data |
| **LightGBM** | Gradient boosting (leaf-wise) | Faster than XGBoost, leaf-wise growth |
| **CatBoost** | Native categorical handling | Best for opponent, weather, position features |
| **Random Forest** | Bagging (decorrelated trees) | Variance reduction, ensemble diversity |

**Ensemble Method:** Weighted averaging using inverse MAE weights from validation set.

**Why 4 models?**
- Diversity: Boosting (XGB, LGBM, CB) + bagging (RF)
- CatBoost handles categoricals natively (opponent, weather conditions)
- Weighted ensemble mitigates individual model weaknesses

### sklearn Pipeline Structure

```python
# For XGBoost, LightGBM, Random Forest
ColumnTransformer([
    ('numeric', 'passthrough', numeric_features),
    ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# CatBoost uses raw categorical data (separate handling)
CatBoostRegressor(cat_features=categorical_features)
```

**Benefits:**
- Prevents data leakage (fit encoders on train only)
- Reproducible transformations (save pipeline with model)
- Easy to add new preprocessing steps

---

## Feature Engineering

### Feature Catalog (~40 features per prop type)

#### Baseline Performance Features (7)
Source: `prop_data_aggregator.py`

1. `weighted_avg_{stat}` - Recency-weighted rolling average (L3: 1.5x, L4-6: 1.0x, L7+: 0.75x)
2. `last_3_avg_{stat}` - Simple L3 average
3. `last_5_avg_{stat}` - Simple L5 average
4. `career_avg_{stat}` - 3-year career per-game average
5. `variance_cv_{stat}` - Coefficient of variation (consistency metric)
6. `games_played` - Sample size indicator
7. `effective_games` - Injury-adjusted effective games

#### Opponent Defense Features (4)
Source: `prop_projection_engine.py` opponent defense calculations

8. `opp_def_pass_ypa` - Opponent pass defense yards/attempt allowed
9. `opp_def_rush_ypc` - Opponent rush defense yards/carry allowed
10. `opp_def_pass_td_rate` - Opponent pass TD rate allowed
11. `opp_def_rush_td_rate` - Opponent rush TD rate allowed

#### Efficiency & Usage Features (4)
Source: `adjustment_pipeline.py`, player stats

12. `success_rate_3wk` - 3-week rolling success rate (chain-moving plays)
13. `success_rate_season` - Season-long success rate
14. `red_zone_rate` - % of touches inside opponent 20
15. `target_share` / `carry_share` - Team usage rate

#### Receiver-Specific Features (4)
Source: `context_adjustments.py`, PBP data

16. `catch_rate` - Completions / targets
17. `catch_rate_over_exp` - Catch rate vs depth-adjusted expected
18. `avg_target_depth` - Average air yards per target
19. `yac_pct` - Yards after catch percentage

#### RB-Specific Features (3)
Source: `context_adjustments.py`

20. `player_ypc` - Player yards per carry
21. `team_ypc` - Teammate RB yards per carry
22. `ypc_diff_pct` - Player YPC / team YPC (OL quality proxy)

#### Route & Field Position Features (3)
Source: `adjustment_pipeline.py` route location

23. `pct_deep_middle` - Deep middle route percentage (15+ yards)
24. `pct_short_sideline` - Short sideline route percentage
25. `slot_rate` - Slot alignment rate (WR/TE)

#### Game Context Features (7)
Source: Schedules, weather data, Vegas lines

26. `is_home` - Home game indicator (0/1)
27. `is_dome` - Dome game indicator (0/1)
28. `division_game` - Division opponent indicator (0/1)
29. `game_temp` - Temperature (°F)
30. `game_wind` - Wind speed (mph)
31. `vegas_total` - Game total over/under
32. `vegas_spread` - Point spread (positive = favorite)

#### Categorical Features (5)
33. `opponent` - Opponent team code (32 values: ARI, ATL, BAL, ...)
34. `position` - Player position (QB, RB, WR, TE)
35. `weather_condition` - Weather description (clear, rain, snow, etc.)
36. `week` - Week of season (1-18)
37. `season` - Year (for time-based trends)

**Total: ~40 features** (varies by prop type - QB passing yards uses all, RB rushing yards excludes receiving features)

### Feature Engineering Implementation

**New file:** `modules/ml_feature_engineering.py`

**Architecture:**
```python
class PropFeatureEngineer:
    """
    Generates ML features for player-week-prop combinations.
    Wraps existing calculation functions from prop_data_aggregator,
    prop_projection_engine, and adjustment_pipeline modules.
    """

    def __init__(self):
        self.context_adj = ContextAdjustments()
        self.pbp_processor = PlayByPlayProcessor()

    def engineer_features(self, player_id, season, week, position, prop_type, opponent_team):
        """
        Generate all features for a player-week-prop combination.

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number
            position: QB/RB/WR/TE
            prop_type: e.g., 'passing_yards'
            opponent_team: Opponent team code

        Returns:
            dict with ~40 features
        """
        features = {}

        # Load player stats through week-1 (no future data)
        player_stats = self._load_player_stats(player_id, season, position)
        stat_col = get_stat_column_for_prop(prop_type)

        # Baseline features (reuse prop_data_aggregator)
        features['weighted_avg'] = calculate_weighted_rolling_average(
            player_stats, stat_col, through_week=week-1
        )
        features['last_3_avg'] = get_simple_average(
            player_stats, stat_col, last_n_games=3, through_week=week-1
        )
        features['last_5_avg'] = get_simple_average(
            player_stats, stat_col, last_n_games=5, through_week=week-1
        )

        career_avgs = get_career_averages(
            player_id, season, position, [stat_col], lookback_years=3
        )
        features['career_avg'] = career_avgs.get(stat_col, 0.0)

        features['variance_cv'] = calculate_stat_variance(
            player_stats, stat_col, through_week=week-1
        )

        features['games_played'] = len(player_stats.filter(pl.col('week') <= week-1))

        # Opponent defense features (extract raw YPA/YPC from adjustment functions)
        opp_def_stats = self._extract_opponent_defense_stats(
            opponent_team, season, week, position, prop_type
        )
        features.update(opp_def_stats)

        # Efficiency features
        features['success_rate_3wk'] = self._calculate_success_rate(
            player_id, season, week, position, prop_type, window=3
        )
        features['success_rate_season'] = self._calculate_success_rate(
            player_id, season, week, position, prop_type, window=None
        )

        # Position-specific features
        if position in ['WR', 'TE', 'RB']:
            catch_features = self._extract_catch_rate_features(
                player_id, season, week
            )
            features.update(catch_features)

        if position == 'RB':
            blocking_features = self._extract_blocking_quality_features(
                player_id, season, week
            )
            features.update(blocking_features)

        # Game context features
        context_features = self._extract_game_context_features(
            player_id, season, week, opponent_team
        )
        features.update(context_features)

        # Categorical features
        features['opponent'] = opponent_team
        features['position'] = position
        features['week'] = week
        features['season'] = season

        return features

    def _extract_opponent_defense_stats(self, opponent, season, week, position, prop_type):
        """Extract raw opponent defensive stats (not multipliers)."""
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
        pbp = pl.read_parquet(pbp_file)
        pbp_filtered = pbp.filter(pl.col('week') < week)  # Through week N-1

        if 'passing' in prop_type:
            defense_stats = pbp_filtered.filter(
                (pl.col('defteam') == opponent) &
                (pl.col('pass_attempt') == 1)
            ).group_by('defteam').agg([
                pl.col('passing_yards').sum().alias('yards_allowed'),
                pl.col('pass_attempt').sum().alias('attempts_faced'),
                pl.col('passing_tds').sum().alias('tds_allowed')
            ])

            if len(defense_stats) > 0:
                ypa = defense_stats['yards_allowed'][0] / max(defense_stats['attempts_faced'][0], 1)
                td_rate = defense_stats['tds_allowed'][0] / max(defense_stats['attempts_faced'][0], 1)
                return {'opp_def_pass_ypa': ypa, 'opp_def_pass_td_rate': td_rate}

        elif 'rushing' in prop_type:
            defense_stats = pbp_filtered.filter(
                (pl.col('defteam') == opponent) &
                (pl.col('rush_attempt') == 1)
            ).group_by('defteam').agg([
                pl.col('rushing_yards').sum().alias('yards_allowed'),
                pl.col('rush_attempt').sum().alias('carries_faced'),
                pl.col('rushing_tds').sum().alias('tds_allowed')
            ])

            if len(defense_stats) > 0:
                ypc = defense_stats['yards_allowed'][0] / max(defense_stats['carries_faced'][0], 1)
                td_rate = defense_stats['tds_allowed'][0] / max(defense_stats['carries_faced'][0], 1)
                return {'opp_def_rush_ypc': ypc, 'opp_def_rush_td_rate': td_rate}

        return {'opp_def_pass_ypa': 7.0, 'opp_def_rush_ypc': 4.3}  # League averages if no data
```

**Key Design Principles:**
1. **No future data:** All features calculated through `week-1`
2. **Reuse existing functions:** Wrap, don't rewrite
3. **Extract raw values:** Not multipliers (ML learns optimal weighting)
4. **Handle missing data:** Return sensible defaults (league averages)

---

## Training Data Generation

### Data Pipeline Architecture

**New file:** `modules/ml_training_data_builder.py`

**Process:**
```
For each year in 2015-2023:
    For each position in [QB, RB, WR, TE]:
        Load positional_player_stats/{position}/{position}-{year}.csv
        For each player-week (week >= 4):
            1. Get opponent from schedule
            2. Generate features using PropFeatureEngineer
            3. Get target = actual stat from same row
            4. Store (features, target) pair

Save to parquet: training_data/{prop_type}_2015_2023.parquet
```

**Implementation:**
```python
def build_training_dataset(start_year=2015, end_year=2023, prop_type='passing_yards'):
    """
    Generate training dataset for a prop type.

    Returns:
        polars DataFrame with feature columns + 'target' column
    """
    feature_engineer = PropFeatureEngineer()
    rows = []

    # Determine eligible positions for this prop type
    prop_config = get_prop_config(prop_type)
    eligible_positions = prop_config['position']

    for year in range(start_year, end_year + 1):
        print(f"Processing {year}...")

        for position in eligible_positions:
            # Load player stats
            stats_file = Path(CACHE_DIR) / f"positional_player_stats/{position.lower()}/{position.lower()}-{year}.csv"
            if not stats_file.exists():
                continue

            stats_df = pl.read_csv(stats_file)
            stat_column = get_stat_column_for_prop(prop_type)

            # Load schedule for opponent matching
            schedule = load_schedule(year)  # Need to generate 2015-2023

            for player_week in stats_df.iter_rows(named=True):
                player_id = player_week['player_id']
                week = player_week['week']
                team = player_week['team']

                # Skip early weeks (need history for features)
                if week < 4:
                    continue

                # Skip if stat is null
                if player_week[stat_column] is None:
                    continue

                # Get opponent from schedule
                opponent = get_opponent_for_week(team, year, week, schedule)
                if opponent is None:
                    continue

                # Generate features
                try:
                    features = feature_engineer.engineer_features(
                        player_id=player_id,
                        season=year,
                        week=week,
                        position=position,
                        prop_type=prop_type,
                        opponent_team=opponent
                    )

                    # Target = actual stat performance
                    features['target'] = player_week[stat_column]
                    rows.append(features)

                except Exception as e:
                    logger.debug(f"Error generating features for {player_id} week {week}: {e}")
                    continue

    # Convert to DataFrame
    train_df = pl.DataFrame(rows)

    # Save to parquet
    output_path = Path(CACHE_DIR) / "ml_training_data" / f"{prop_type}_2015_2023.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(output_path)

    print(f"\n=== Training Data Summary ===")
    print(f"Prop type: {prop_type}")
    print(f"Total examples: {len(train_df)}")
    print(f"Features: {len(train_df.columns) - 1}")  # Exclude 'target'
    print(f"Saved to: {output_path}")

    return train_df
```

### Training Data Size Estimates

| Prop Type | Positions | Starters | Weeks | Years | Est. Examples |
|-----------|-----------|----------|-------|-------|---------------|
| passing_yards | QB | 32 | 14 | 9 | ~4,000 |
| passing_tds | QB | 32 | 14 | 9 | ~4,000 |
| rushing_yards | QB+RB | 64 | 14 | 9 | ~8,000 |
| rushing_tds | QB+RB | 64 | 14 | 9 | ~8,000 |
| receptions | RB+WR+TE | 160 | 14 | 9 | ~20,000 |
| receiving_yards | RB+WR+TE | 160 | 14 | 9 | ~20,000 |
| receiving_tds | RB+WR+TE | 160 | 14 | 9 | ~20,000 |

**Total across all prop types: ~84,000 training examples**

**Data Quality Checks:**
- Remove rows with null targets
- Remove rows with <3 games played (insufficient feature history)
- Validate feature ranges (no negative yards, success rate 0-1, etc.)
- Check for data leakage (no future-looking features)

---

## Model Training Implementation

### Ensemble Training Pipeline

**New file:** `modules/ml_ensemble.py`

**Class Structure:**
```python
class PropEnsembleModel:
    """
    4-model ensemble for prop prediction.

    Models:
    - XGBoost (sklearn Pipeline with OneHotEncoder)
    - LightGBM (sklearn Pipeline with OneHotEncoder)
    - Random Forest (sklearn Pipeline with OneHotEncoder)
    - CatBoost (native categorical handling)

    Ensemble: Weighted averaging with inverse MAE weights
    """

    def __init__(self, prop_type, numeric_features, categorical_features):
        self.prop_type = prop_type
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.models = {}
        self.weights = {}
        self.feature_importances = {}

    def build_pipelines(self):
        """Build sklearn pipelines for each model."""

        # Preprocessing for tree models
        tree_preprocessor = ColumnTransformer([
            ('num', 'passthrough', self.numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             self.categorical_features)
        ])

        # XGBoost pipeline
        self.models['xgboost'] = Pipeline([
            ('preprocessor', tree_preprocessor),
            ('model', xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ))
        ])

        # LightGBM pipeline
        self.models['lightgbm'] = Pipeline([
            ('preprocessor', tree_preprocessor),
            ('model', lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42
            ))
        ])

        # Random Forest pipeline
        self.models['random_forest'] = Pipeline([
            ('preprocessor', tree_preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ))
        ])

        # CatBoost (uses raw categorical data)
        cat_feature_indices = [self.numeric_features.index(f) if f in self.numeric_features
                              else len(self.numeric_features) + self.categorical_features.index(f)
                              for f in self.categorical_features]

        self.models['catboost'] = CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function='RMSE',
            cat_features=cat_feature_indices,
            verbose=False,
            random_state=42
        )

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train all 4 models and calculate optimal ensemble weights.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (for weight calculation)

        Returns:
            dict with validation metrics for each model
        """
        val_predictions = {}
        val_metrics = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            if name == 'catboost':
                # CatBoost handles categoricals natively
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                val_pred = model.predict(X_val)
            else:
                # sklearn pipelines
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)

            # Store predictions
            val_predictions[name] = val_pred

            # Calculate metrics
            mae = np.mean(np.abs(val_pred - y_val))
            mse = np.mean((val_pred - y_val) ** 2)
            rmse = np.sqrt(mse)

            val_metrics[name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse
            }

            print(f"  Validation MAE: {mae:.2f}")
            print(f"  Validation RMSE: {rmse:.2f}")

        # Calculate inverse MAE weights
        total_inv_mae = sum(1/metrics['mae'] for metrics in val_metrics.values())
        self.weights = {
            name: (1/metrics['mae']) / total_inv_mae
            for name, metrics in val_metrics.items()
        }

        print(f"\n=== Ensemble Weights (inverse MAE) ===")
        for name, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            print(f"  {name}: {weight:.3f}")

        return val_metrics

    def predict(self, X):
        """
        Generate weighted ensemble prediction.

        Args:
            X: Feature DataFrame

        Returns:
            np.array of predictions
        """
        predictions = {}

        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        # Weighted average
        ensemble_pred = np.sum([
            self.weights[name] * pred
            for name, pred in predictions.items()
        ], axis=0)

        return ensemble_pred

    def save(self, filepath):
        """Save model ensemble to disk."""
        import joblib
        joblib.dump({
            'models': self.models,
            'weights': self.weights,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'prop_type': self.prop_type
        }, filepath)

    @staticmethod
    def load(filepath):
        """Load model ensemble from disk."""
        import joblib
        data = joblib.load(filepath)

        ensemble = PropEnsembleModel(
            data['prop_type'],
            data['numeric_features'],
            data['categorical_features']
        )
        ensemble.models = data['models']
        ensemble.weights = data['weights']

        return ensemble
```

### Hyperparameter Configuration

**Starting Point (will tune based on validation):**

```python
HYPERPARAMETERS = {
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 12,
        'min_samples_split': 10,
        'min_samples_leaf': 5
    },
    'catboost': {
        'iterations': 200,
        'depth': 6,
        'learning_rate': 0.05
    }
}
```

**Rationale:**
- **n_estimators=200:** Sufficient for convergence without overfitting
- **max_depth=6-12:** Shallow enough to generalize, deep enough to capture interactions
- **learning_rate=0.05:** Conservative to prevent overfitting
- **subsample/colsample:** Regularization through subsampling

**Tuning Strategy:** Grid search over validation set in Phase 5 if initial results underwhelming.

---

## Cross-Validation Strategy

### Time-Series Aware Validation

**Use sklearn TimeSeriesSplit** to preserve temporal ordering:

```python
from sklearn.model_selection import TimeSeriesSplit

def cross_validate_model(train_df, prop_type, n_splits=5):
    """
    Perform time-series cross-validation.

    Args:
        train_df: Training DataFrame (2015-2023)
        prop_type: Prop type to train
        n_splits: Number of folds (default 5)

    Returns:
        dict with cross-validation results
    """
    # Separate features and target
    feature_cols = [col for col in train_df.columns if col != 'target']
    X = train_df.select(feature_cols).to_pandas()
    y = train_df['target'].to_pandas()

    # Time-series split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_results = {
        'fold_metrics': [],
        'mean_mae': 0,
        'mean_mse': 0,
        'mean_pct_correct': 0,
        'std_pct_correct': 0
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Build and train model
        numeric_features = [col for col in feature_cols if col not in ['opponent', 'position', 'weather_condition']]
        categorical_features = ['opponent', 'position', 'weather_condition']

        ensemble = PropEnsembleModel(prop_type, numeric_features, categorical_features)
        ensemble.build_pipelines()
        ensemble.train(X_train, y_train, X_val, y_val)

        # Predict on validation
        y_pred = ensemble.predict(X_val)

        # Calculate metrics
        fold_metrics = calculate_metrics(y_val, y_pred, betting_lines=None)
        cv_results['fold_metrics'].append(fold_metrics)

        print(f"\nFold {fold+1} Results:")
        print(f"  MAE: {fold_metrics['mae']:.2f}")
        print(f"  MSE: {fold_metrics['mse']:.2f}")
        print(f"  RMSE: {fold_metrics['rmse']:.2f}")

    # Aggregate across folds
    cv_results['mean_mae'] = np.mean([m['mae'] for m in cv_results['fold_metrics']])
    cv_results['mean_mse'] = np.mean([m['mse'] for m in cv_results['fold_metrics']])
    cv_results['std_mae'] = np.std([m['mae'] for m in cv_results['fold_metrics']])

    return cv_results
```

**Why Time-Series Split:**
- Preserves temporal ordering (train on past, validate on future)
- Prevents data leakage from future to past
- Simulates real-world deployment (always predicting future weeks)

### Per-Prop-Category Validation

**Validate each of 7 prop types independently:**

```python
prop_types = ['passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
              'receptions', 'receiving_yards', 'receiving_tds']

all_results = {}

for prop_type in prop_types:
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATING: {prop_type}")
    print(f"{'='*60}")

    # Load training data
    train_df = pl.read_parquet(
        Path(CACHE_DIR) / "ml_training_data" / f"{prop_type}_2015_2023.parquet"
    )

    # Cross-validate
    cv_results = cross_validate_model(train_df, prop_type, n_splits=5)
    all_results[prop_type] = cv_results

    # Summary
    print(f"\n=== {prop_type} Cross-Validation Summary ===")
    print(f"Average MAE: {cv_results['mean_mae']:.2f} ± {cv_results['std_mae']:.2f}")
    print(f"Average MSE: {cv_results['mean_mse']:.2f}")

# Save results
import json
with open('research/ML_CROSS_VALIDATION_RESULTS.json', 'w') as f:
    json.dump(all_results, f, indent=2)
```

---

## Evaluation Metrics

### Three Required Metrics

#### 1. Percent Correct (%correct)

**Definition:** Percentage of predictions that correctly identified OVER/UNDER vs betting line.

**Implementation:**
```python
def calculate_percent_correct(y_true, y_pred, betting_lines):
    """
    Calculate % of predictions that correctly predicted OVER/UNDER vs line.

    Args:
        y_true: Actual stat outcomes (np.array)
        y_pred: Model predictions (np.array)
        betting_lines: Betting line values (np.array, may contain None)

    Returns:
        float: Percentage correct (0-100)
    """
    correct = 0
    total = 0

    for true, pred, line in zip(y_true, y_pred, betting_lines):
        if line is not None and not np.isnan(line):
            # Actual result
            actual_over = true > line

            # Prediction
            pred_over = pred > line

            # Check if correct
            if actual_over == pred_over:
                correct += 1

            total += 1

    return (correct / total * 100) if total > 0 else 0.0
```

**Break-Even:** 52.4% at -110 odds (need to win 52.4% to overcome vig)

#### 2. Mean Absolute Error (MAE)

**Definition:** Average absolute difference between prediction and actual.

**Implementation:**
```python
def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        float: MAE
    """
    return np.mean(np.abs(y_true - y_pred))
```

**Target:** <30 yards for passing/receiving, <15 yards for rushing

#### 3. Mean Squared Error (MSE)

**Definition:** Average squared difference between prediction and actual.

**Implementation:**
```python
def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        float: MSE
    """
    return np.mean((y_true - y_pred) ** 2)
```

**Derived Metric:** RMSE = sqrt(MSE) for interpretability

### Comprehensive Metrics Function

```python
def calculate_metrics(y_true, y_pred, betting_lines=None):
    """
    Calculate all evaluation metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        betting_lines: Optional betting line values for %correct

    Returns:
        dict with metrics
    """
    metrics = {
        'mae': calculate_mae(y_true, y_pred),
        'mse': calculate_mse(y_true, y_pred),
        'rmse': np.sqrt(calculate_mse(y_true, y_pred))
    }

    if betting_lines is not None:
        metrics['pct_correct'] = calculate_percent_correct(y_true, y_pred, betting_lines)

    return metrics
```

### Validation Output Format

```
=== Cross-Validation Results: passing_yards ===

Fold 1 (2015-2017 train, 2018 val):
  MAE: 32.4 yards
  MSE: 1,847.2
  RMSE: 42.9 yards
  %Correct: 56.2% (vs 52.4% break-even)

Fold 2 (2015-2018 train, 2019 val):
  MAE: 34.1 yards
  MSE: 1,923.5
  RMSE: 43.9 yards
  %Correct: 54.8%

Fold 3 (2015-2019 train, 2020 val):
  MAE: 31.8 yards
  MSE: 1,802.1
  RMSE: 42.5 yards
  %Correct: 55.9%

Fold 4 (2015-2020 train, 2021 val):
  MAE: 33.5 yards
  MSE: 1,876.3
  RMSE: 43.3 yards
  %Correct: 54.2%

Fold 5 (2015-2021 train, 2022 val):
  MAE: 32.9 yards
  MSE: 1,891.7
  RMSE: 43.5 yards
  %Correct: 55.6%

=== Average Across All Folds ===
  MAE: 33.0 ± 1.0 yards
  MSE: 1,868.2 ± 48.7
  RMSE: 43.2 ± 0.6 yards
  %Correct: 55.3% ± 0.8%

=== Feature Importances (XGBoost) ===
  1. weighted_avg: 0.352
  2. last_3_avg: 0.118
  3. opp_def_pass_ypa: 0.095
  4. success_rate_3wk: 0.073
  5. career_avg: 0.061
  6. variance_cv: 0.048
  7. games_played: 0.042
  8. opponent_BUF: 0.031
  9. opponent_KC: 0.028
  10. last_5_avg: 0.027
  ... (top 20 shown)
```

---

## Testing Implementation

### Test Directory Structure

```
tests/ml/
├── __init__.py
├── fixtures/
│   ├── sample_training_data.csv       # 100 rows for fast tests
│   ├── sample_player_stats.csv        # QB stats 2023
│   ├── sample_pbp_data.parquet        # PBP sample
│   ├── expected_features.json         # Known feature values
│   └── test_model.pkl                 # Pre-trained small model
│
├── test_feature_engineering.py        # Test feature extraction
├── test_training_data_builder.py      # Test dataset generation
├── test_model_training.py             # Test model training
├── test_model_predictions.py          # Test prediction generation
├── test_cross_validation.py           # Test CV implementation
└── test_ml_backtest.py                # End-to-end backtest
```

### Test Patterns

#### Test 1: Feature Engineering

**File:** `tests/ml/test_feature_engineering.py`

```python
import pytest
from modules.ml_feature_engineering import PropFeatureEngineer

def test_feature_extraction_qb_passing_yards():
    """Test feature engineering for QB passing yards prop."""
    engineer = PropFeatureEngineer()

    # Patrick Mahomes 2024 Week 10
    features = engineer.engineer_features(
        player_id="00-0033873",
        season=2024,
        week=10,
        position='QB',
        prop_type='passing_yards',
        opponent_team='BUF'
    )

    # Validate feature presence
    required_features = [
        'weighted_avg', 'last_3_avg', 'last_5_avg', 'career_avg',
        'variance_cv', 'games_played', 'opp_def_pass_ypa',
        'success_rate_3wk', 'opponent', 'position'
    ]
    for feature in required_features:
        assert feature in features, f"Missing feature: {feature}"

    # Validate feature ranges
    assert 200 <= features['weighted_avg'] <= 400, "Weighted avg out of range"
    assert 0 <= features['success_rate_3wk'] <= 1, "Success rate out of range"
    assert features['opp_def_pass_ypa'] > 0, "Opponent defense should be positive"
    assert features['games_played'] >= 1, "Games played should be >= 1"

    # Validate categorical features
    assert features['opponent'] == 'BUF'
    assert features['position'] == 'QB'

def test_feature_extraction_handles_missing_data():
    """Test that feature engineering handles missing data gracefully."""
    engineer = PropFeatureEngineer()

    # Early-career player with limited history
    features = engineer.engineer_features(
        player_id="00-0012345",  # Fictional rookie
        season=2024,
        week=5,
        position='QB',
        prop_type='passing_yards',
        opponent_team='KC'
    )

    # Should return defaults, not crash
    assert 'weighted_avg' in features
    assert features['career_avg'] >= 0  # May be 0 if no career data

def test_feature_extraction_no_future_data():
    """Test that features only use data through week-1."""
    engineer = PropFeatureEngineer()

    # Week 10 features should only use weeks 1-9 data
    features_week10 = engineer.engineer_features(
        player_id="00-0033873",
        season=2024,
        week=10,
        position='QB',
        prop_type='passing_yards',
        opponent_team='BUF'
    )

    # Week 11 features should only use weeks 1-10 data
    features_week11 = engineer.engineer_features(
        player_id="00-0033873",
        season=2024,
        week=11,
        position='QB',
        prop_type='passing_yards',
        opponent_team='CAR'
    )

    # Week 11 weighted avg should be >= week 10 (more data)
    assert features_week11['games_played'] > features_week10['games_played']
```

#### Test 2: Model Training

**File:** `tests/ml/test_model_training.py`

```python
import pytest
import polars as pl
from modules.ml_ensemble import PropEnsembleModel

def test_model_training_basic():
    """Test basic model training workflow."""
    # Load fixture data
    train_df = pl.read_csv('tests/ml/fixtures/sample_training_data.csv')

    # Split into train/val
    split_idx = int(len(train_df) * 0.8)
    train = train_df[:split_idx]
    val = train_df[split_idx:]

    feature_cols = [col for col in train.columns if col != 'target']
    X_train, y_train = train.select(feature_cols).to_pandas(), train['target'].to_pandas()
    X_val, y_val = val.select(feature_cols).to_pandas(), val['target'].to_pandas()

    # Build ensemble
    numeric_features = [col for col in feature_cols if col not in ['opponent', 'position']]
    categorical_features = ['opponent', 'position']

    ensemble = PropEnsembleModel('passing_yards', numeric_features, categorical_features)
    ensemble.build_pipelines()

    # Train
    metrics = ensemble.train(X_train, y_train, X_val, y_val)

    # Validate models trained
    assert len(ensemble.models) == 4
    assert 'xgboost' in ensemble.models
    assert 'lightgbm' in ensemble.models
    assert 'catboost' in ensemble.models
    assert 'random_forest' in ensemble.models

    # Validate weights calculated
    assert len(ensemble.weights) == 4
    assert sum(ensemble.weights.values()) == pytest.approx(1.0, rel=1e-2)

    # Validate metrics reasonable
    for name, model_metrics in metrics.items():
        assert model_metrics['mae'] > 0
        assert model_metrics['mse'] > 0

def test_model_prediction():
    """Test model prediction workflow."""
    # Load pre-trained model
    import joblib
    ensemble = joblib.load('tests/ml/fixtures/test_model.pkl')

    # Load test data
    test_df = pl.read_csv('tests/ml/fixtures/sample_training_data.csv')
    feature_cols = [col for col in test_df.columns if col != 'target']
    X_test = test_df.select(feature_cols).to_pandas()

    # Predict
    predictions = ensemble.predict(X_test)

    # Validate predictions
    assert len(predictions) == len(X_test)
    assert all(pred > 0 for pred in predictions)  # No negative yards
    assert all(pred < 600 for pred in predictions)  # No unrealistic values

def test_model_save_load():
    """Test model persistence."""
    import tempfile

    # Create and train small model
    train_df = pl.read_csv('tests/ml/fixtures/sample_training_data.csv')
    feature_cols = [col for col in train_df.columns if col != 'target']
    X = train_df.select(feature_cols).to_pandas()
    y = train_df['target'].to_pandas()

    numeric_features = [col for col in feature_cols if col not in ['opponent', 'position']]
    categorical_features = ['opponent', 'position']

    ensemble = PropEnsembleModel('passing_yards', numeric_features, categorical_features)
    ensemble.build_pipelines()
    split_idx = int(len(X) * 0.8)
    ensemble.train(X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:])

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        ensemble.save(f.name)
        temp_path = f.name

    # Load
    loaded_ensemble = PropEnsembleModel.load(temp_path)

    # Validate loaded correctly
    assert loaded_ensemble.prop_type == 'passing_yards'
    assert len(loaded_ensemble.models) == 4
    assert len(loaded_ensemble.weights) == 4

    # Validate predictions match
    pred_original = ensemble.predict(X[:10])
    pred_loaded = loaded_ensemble.predict(X[:10])
    assert np.allclose(pred_original, pred_loaded)
```

#### Test 3: End-to-End Backtest

**File:** `tests/ml/test_ml_backtest.py`

```python
def test_ml_backtest_2024_week5():
    """Test ML predictions vs actual outcomes for 2024 Week 5."""
    from modules.ml_ensemble import PropEnsembleModel
    from modules.ml_feature_engineering import PropFeatureEngineer

    # Load trained model (assume pre-trained for test)
    ensemble = PropEnsembleModel.load('models/passing_yards_ensemble.pkl')
    engineer = PropFeatureEngineer()

    # Get Week 5 QB players
    players = get_week_players(2024, 5, 'QB')

    predictions = []
    actuals = []

    for player in players:
        # Generate features
        features = engineer.engineer_features(
            player_id=player.id,
            season=2024,
            week=5,
            position='QB',
            prop_type='passing_yards',
            opponent_team=player.opponent
        )

        # Predict
        pred = ensemble.predict(pd.DataFrame([features]))[0]
        predictions.append(pred)

        # Get actual outcome
        actual = get_actual_outcome(player.id, 2024, 5, 'passing_yards')
        actuals.append(actual)

    # Calculate metrics
    mae = calculate_mae(actuals, predictions)
    mse = calculate_mse(actuals, predictions)

    # Assertions
    assert mae < 40, f"MAE {mae} exceeds threshold"
    assert len(predictions) >= 20, "Should have predictions for at least 20 QBs"

    # If betting lines available, test %correct
    betting_lines = load_betting_lines(2024, 5, 'passing_yards')
    if betting_lines:
        pct_correct = calculate_percent_correct(actuals, predictions, betting_lines)
        assert pct_correct > 50, f"%Correct {pct_correct}% should beat coin flip"
```

### Test Execution

```bash
# Run all ML tests
python -m pytest tests/ml/ -v

# Run specific test file
python -m pytest tests/ml/test_feature_engineering.py -v

# Run with coverage
python -m pytest tests/ml/ --cov=modules.ml_feature_engineering --cov-report=html
```

---

## Implementation Timeline

### Phase 1: Feature Engineering (Week 1, Days 1-3)

**Tasks:**
- [ ] Create `modules/ml_feature_engineering.py`
- [ ] Implement `PropFeatureEngineer` class
- [ ] Wrap existing calculation functions (prop_data_aggregator, prop_projection_engine)
- [ ] Create `tests/ml/test_feature_engineering.py`
- [ ] Test on sample players (Mahomes, CMC, Tyreek Hill)
- [ ] Document feature catalog

**Deliverables:**
- Working feature engineering module
- Unit tests passing
- Feature documentation

**Success Criteria:**
- Extract ~40 features for passing_yards prop
- All features in valid ranges
- No future data leakage

---

### Phase 2: Training Data Generation (Week 1, Days 4-7)

**Tasks:**
- [ ] Create `modules/ml_training_data_builder.py`
- [ ] Generate schedules for 2015-2023 (opponent matching)
- [ ] Build training datasets for all 7 prop types
- [ ] Save as parquet files (`cache/ml_training_data/{prop_type}_2015_2023.parquet`)
- [ ] Validate data quality (no nulls, correct ranges)
- [ ] Create data summary reports

**Deliverables:**
- 7 training datasets (one per prop type)
- ~84,000 total training examples
- Data quality validation report

**Success Criteria:**
- All 7 datasets generated successfully
- <5% null rate for features
- All features in expected ranges

---

### Phase 3: Model Training Infrastructure (Week 2, Days 1-4)

**Tasks:**
- [ ] Create `modules/ml_ensemble.py`
- [ ] Implement `PropEnsembleModel` class
- [ ] Build sklearn Pipelines for XGBoost, LightGBM, Random Forest
- [ ] Integrate CatBoost with native categorical handling
- [ ] Implement weighted ensemble averaging
- [ ] Implement cross-validation with TimeSeriesSplit
- [ ] Create `tests/ml/test_model_training.py`
- [ ] Test on small dataset

**Deliverables:**
- Complete training framework
- Cross-validation implementation
- Model save/load functionality

**Success Criteria:**
- Train on sample data without errors
- Cross-validation runs successfully
- Models save/load correctly

---

### Phase 4: Train Initial Models (Week 2, Days 5-7)

**Tasks:**
- [ ] Train `passing_yards` model (QB)
- [ ] Train `receiving_yards` model (WR/TE/RB)
- [ ] Run 5-fold cross-validation for both
- [ ] Calculate MAE, MSE, %correct metrics
- [ ] Generate feature importance reports
- [ ] Validate on 2024 data (holdout set)
- [ ] Compare vs rule-based system baseline

**Deliverables:**
- 2 trained models (passing_yards, receiving_yards)
- Cross-validation results
- Feature importance analysis
- Performance comparison report

**Success Criteria:**
- MAE < 35 yards for both models
- %Correct > 52% on validation
- Beats rule-based baseline

---

### Phase 5: Complete Model Suite (Week 3, Days 1-4)

**Tasks:**
- [ ] Train remaining 5 prop type models:
  - passing_tds
  - rushing_yards
  - rushing_tds
  - receptions
  - receiving_tds
- [ ] Run cross-validation for all 5
- [ ] Optimize hyperparameters based on validation results
- [ ] Generate comprehensive feature importance reports
- [ ] Save all trained models to `models/` directory

**Deliverables:**
- All 7 prop type models trained and validated
- Hyperparameter tuning report
- Complete feature importance catalog

**Success Criteria:**
- All models train successfully
- Average %correct > 52% across all prop types
- No single prop type below 48%

---

### Phase 6: Integration & Backtesting (Week 3, Days 5-7)

**Tasks:**
- [ ] Modify `modules/prop_evaluator.py` to use ML predictions
- [ ] Update `backtest_props.py` for ML system
- [ ] Run full backtest on 2024 data (weeks 5-10)
- [ ] Calculate metrics: %correct, MAE, MSE, ROI
- [ ] Compare ML vs rule-based performance side-by-side
- [ ] Generate comprehensive comparison report
- [ ] Document findings in `research/ML_VS_RULEBASED_COMPARISON.md`

**Deliverables:**
- ML system integrated into backtesting
- 2024 backtest results
- Performance comparison report

**Success Criteria:**
- ML system runs without errors on 2024 data
- %Correct > rule-based (37-38%)
- ROI improvement demonstrated

---

### Phase 7: Production Deployment (Week 4)

**Tasks:**
- [ ] Create `props_ml.py` CLI for ML predictions
- [ ] Add model versioning (v1.0.0)
- [ ] Create model retraining script
- [ ] Set up model rollback capability
- [ ] Deploy for live betting evaluation
- [ ] Monitor initial live performance
- [ ] Document deployment procedures

**Deliverables:**
- Production-ready ML props system
- Deployment documentation
- Monitoring dashboard

**Success Criteria:**
- System generates predictions for current week
- Model versioning works correctly
- Rollback tested successfully

---

## Success Criteria

### Minimum Viable Performance (MVP)

**Cross-Validation Metrics:**
- **%Correct:** >52% average across all folds (beat 50% coin flip)
- **MAE:** <35 yards for passing/receiving, <18 yards for rushing
- **Stability:** Standard deviation <2 pp across folds

**Backtest Metrics (2024 weeks 5-10):**
- **%Correct:** >50% (beat coin flip)
- **MAE:** Comparable to or better than rule-based (24-34 yards)
- **No catastrophic failures:** No prop type below 45%

---

### Target Performance (Production Ready)

**Cross-Validation Metrics:**
- **%Correct:** >55% average (beat 52.4% break-even by 2.6+ pp)
- **MAE:** <30 yards for passing/receiving, <15 yards for rushing
- **Consistency:** All prop types within 3 pp of average

**Backtest Metrics (2024 weeks 5-10):**
- **%Correct:** >55%
- **ROI:** >+5% with realistic bet sizing
- **Improvement:** Win rate 10+ pp better than rule-based (37-38%)

**Feature Importance:**
- Top 3 features account for <60% of importance (diversified)
- Baseline features dominate (weighted_avg, last_3_avg, career_avg)
- Context features show measurable impact (opponent defense, success rate)

---

### Stretch Goals (Elite Performance)

**Cross-Validation Metrics:**
- **%Correct:** >58% average
- **MAE:** <25 yards for passing/receiving, <12 yards for rushing
- **Consistency:** All prop types within 2 pp of average

**Backtest Metrics:**
- **%Correct:** >58%
- **ROI:** >+10%
- **Sharpe Ratio:** >0.5 (risk-adjusted returns)

**Market Impact:**
- Demonstrate edge on 2023 data as well (out-of-sample validation)
- Show consistent performance across different bet sizes
- Identify specific player types / game situations with highest edge

---

### Reality Check & Decision Points

**After Phase 4 (Initial Models):**

If %Correct < 50%:
- **Action:** Revisit feature engineering, add more contextual features
- **Timeline:** +1 week for iteration

If %Correct 50-52%:
- **Action:** Proceed but with caution, may not be profitable
- **Timeline:** Continue to Phase 5

If %Correct > 52%:
- **Action:** Proceed confidently
- **Timeline:** On track

**After Phase 6 (Full Backtest):**

If %Correct < 48%:
- **Decision:** Market likely unbeatable with public data
- **Action:** Document findings, accept limitations

If %Correct 48-52%:
- **Decision:** Close but no consistent edge
- **Action:** Consider niche strategies (early season, backup players)

If %Correct > 52%:
- **Decision:** Viable edge exists
- **Action:** Proceed to production with strict bankroll management

**Honest Assessment Commitment:**
- If final %Correct < 48%, acknowledge that betting lines are too efficient
- Document why ML approach didn't overcome market efficiency
- Pivot to learning experience rather than continue unprofitable path

---

## File Changes Summary

### Files to Keep (Unmodified)
- `modules/prop_types.py` - Prop type configuration
- `modules/prop_data_aggregator.py` - Feature calculations (all reused)
- `modules/prop_projection_engine.py` - Adjustment calculations (extract raw stats)
- `modules/adjustment_pipeline.py` - Batch calculations (reused for features)
- `modules/context_adjustments.py` - Context metrics (reused for features)
- `modules/prop_output_formatter.py` - Output formatting
- `modules/constants.py` - Constants
- All data loading infrastructure

### Files to Modify (Minimal Changes)
- `modules/prop_evaluator.py` - Replace `generate_projection()` call with ML `predict()` (1 line)
- `backtest_props.py` - Same replacement (1 line)

### Files to Create (New)
- `modules/ml_feature_engineering.py` (~300 lines) - Feature extraction wrapper
- `modules/ml_training_data_builder.py` (~200 lines) - Dataset generation
- `modules/ml_ensemble.py` (~400 lines) - 4-model ensemble with sklearn Pipeline
- `modules/ml_model.py` (~200 lines) - Model management utilities
- `scripts/train_models.py` (~150 lines) - Training CLI
- `tests/ml/test_feature_engineering.py` (~200 lines) - Feature tests
- `tests/ml/test_model_training.py` (~150 lines) - Training tests
- `tests/ml/test_ml_backtest.py` (~100 lines) - Backtest tests
- `research/ML_IMPLEMENTATION_PLAN.md` - This document

**Total New Code:** ~1,700 lines
**Code Reused:** ~1,000 lines from existing modules
**Efficiency:** 63% code reuse, 37% new code

---

## Risk Mitigation

### Risk 1: Overfitting to Training Data

**Symptoms:**
- High training accuracy, low validation accuracy
- Large gap between cross-validation and backtest performance
- Model memorizes specific player-week patterns

**Mitigation:**
- Time-series cross-validation (train on past, validate on future)
- Hold out 2024 as separate test set (never used in training)
- Regularization in tree models (max_depth, min_samples_leaf)
- Ensemble diversity (boosting + bagging)

**Monitoring:**
- Track train vs validation MAE gap
- If gap > 10 yards, reduce model complexity

---

### Risk 2: Feature Leakage (Future Data in Features)

**Symptoms:**
- Unrealistically high validation accuracy
- Performance degrades sharply in production
- Features contain information not available at prediction time

**Mitigation:**
- Strict `through_week` filtering in all feature calculations
- Test suite validates no future data in features
- Manual review of feature engineering logic

**Monitoring:**
- Compare week N features vs week N+1 features (should only differ by 1 game)
- Verify opponent defense uses weeks 1 through N-1 only

---

### Risk 3: Insufficient Training Data

**Symptoms:**
- High variance across cross-validation folds
- Poor performance on rare player types (backup QBs, TEs)
- Models fail to generalize

**Mitigation:**
- 9 years of data (2015-2023) = ~84,000 examples
- Start with high-volume props (receiving_yards, passing_yards)
- Pool positions if needed (all receivers together)
- Minimum 3 games played filter

**Monitoring:**
- Check sample size per position-prop combination
- If <2,000 examples, consider pooling positions

---

### Risk 4: Market Efficiency (Lines Too Sharp)

**Symptoms:**
- %Correct hovers around 50% despite sophisticated model
- No consistent edge across prop types
- Random walk-like performance

**Reality:**
- Betting lines are set by professionals using similar (or better) models
- Public data may not provide edge over market
- Sharp bettors have private information (injury reports, lineup intel)

**Mitigation:**
- Set realistic expectations (%Correct > 52% is success)
- Honest assessment after Phase 6 backtest
- If %Correct < 48%, accept market is unbeatable

**Decision Point:**
- After Phase 6, make go/no-go decision based on actual results
- If unprofitable, document findings and pivot to learning experience

---

## Next Immediate Steps

### Step 1: Create Feature Engineering Module (Days 1-2)
1. Create `modules/ml_feature_engineering.py`
2. Implement `PropFeatureEngineer` class
3. Extract baseline features from `prop_data_aggregator.py`
4. Extract opponent defense from `prop_projection_engine.py`
5. Test on Patrick Mahomes 2024 Week 10

### Step 2: Generate Sample Training Data (Day 3)
1. Create `modules/ml_training_data_builder.py`
2. Generate 2023 passing_yards dataset (test on 1 year)
3. Validate data quality
4. If successful, generate 2015-2023 full dataset

### Step 3: Train Proof-of-Concept Model (Days 4-5)
1. Create `modules/ml_ensemble.py`
2. Train on 2023 passing_yards only
3. Validate on 2024 weeks 5-10
4. Compare vs rule-based baseline

### Step 4: Iterate Based on Results (Days 6-7)
1. Analyze feature importances
2. Adjust features if needed
3. Tune hyperparameters
4. Re-validate

**Go/No-Go Decision Point:** After Step 4, if %Correct < 50%, revisit approach before scaling to all prop types.

---

## Appendix A: Feature Extraction Code Examples

### Example 1: Baseline Features

```python
def _extract_baseline_features(self, player_id, season, week, position, prop_type):
    """Extract baseline performance features."""
    # Load player stats
    stats_file = Path(CACHE_DIR) / f"positional_player_stats/{position.lower()}/{position.lower()}-{season}.csv"
    player_stats = pl.read_csv(stats_file).filter(pl.col('player_id') == player_id)

    stat_col = get_stat_column_for_prop(prop_type)

    features = {}

    # Weighted rolling average (recency-weighted)
    features['weighted_avg'] = calculate_weighted_rolling_average(
        player_stats, stat_col, through_week=week-1
    )

    # Simple averages
    features['last_3_avg'] = get_simple_average(
        player_stats, stat_col, last_n_games=3, through_week=week-1
    )
    features['last_5_avg'] = get_simple_average(
        player_stats, stat_col, last_n_games=5, through_week=week-1
    )

    # Career average (3-year lookback)
    career_avgs = get_career_averages(
        player_id, season, position, [stat_col], lookback_years=3
    )
    features['career_avg'] = career_avgs.get(stat_col, 0.0)

    # Variance (consistency metric)
    features['variance_cv'] = calculate_stat_variance(
        player_stats, stat_col, through_week=week-1
    )

    # Sample size
    features['games_played'] = len(player_stats.filter(pl.col('week') <= week-1))

    return features
```

### Example 2: Opponent Defense Features

```python
def _extract_opponent_defense_features(self, opponent, season, week, position, prop_type):
    """Extract raw opponent defensive stats."""
    pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{season}.parquet"
    pbp = pl.read_parquet(pbp_file)

    # Filter to weeks 1 through N-1 (no future data)
    pbp_filtered = pbp.filter(pl.col('week') < week)

    features = {}

    if 'passing' in prop_type:
        # Opponent pass defense
        defense = pbp_filtered.filter(
            (pl.col('defteam') == opponent) &
            (pl.col('pass_attempt') == 1)
        ).group_by('defteam').agg([
            pl.col('passing_yards').sum().alias('yards_allowed'),
            pl.col('pass_attempt').sum().alias('attempts'),
            pl.col('passing_tds').sum().alias('tds_allowed')
        ])

        if len(defense) > 0:
            features['opp_def_pass_ypa'] = defense['yards_allowed'][0] / max(defense['attempts'][0], 1)
            features['opp_def_pass_td_rate'] = defense['tds_allowed'][0] / max(defense['attempts'][0], 1)
        else:
            features['opp_def_pass_ypa'] = 7.0  # League average
            features['opp_def_pass_td_rate'] = 0.045  # ~4.5%

    elif 'rushing' in prop_type:
        # Opponent rush defense
        defense = pbp_filtered.filter(
            (pl.col('defteam') == opponent) &
            (pl.col('rush_attempt') == 1)
        ).group_by('defteam').agg([
            pl.col('rushing_yards').sum().alias('yards_allowed'),
            pl.col('rush_attempt').sum().alias('carries'),
            pl.col('rushing_tds').sum().alias('tds_allowed')
        ])

        if len(defense) > 0:
            features['opp_def_rush_ypc'] = defense['yards_allowed'][0] / max(defense['carries'][0], 1)
            features['opp_def_rush_td_rate'] = defense['tds_allowed'][0] / max(defense['carries'][0], 1)
        else:
            features['opp_def_rush_ypc'] = 4.3  # League average
            features['opp_def_rush_td_rate'] = 0.018  # ~1.8%

    return features
```

---

## Appendix B: Cross-Validation Example Output

```
=== Cross-Validation: passing_yards ===
Training data: 4,127 examples (2015-2023)

Fold 1/5 (Train: 2015-2017, Val: 2018)
  Training XGBoost...     Val MAE: 31.2
  Training LightGBM...    Val MAE: 32.1
  Training Random Forest...Val MAE: 33.5
  Training CatBoost...    Val MAE: 30.8

  Ensemble Weights:
    CatBoost: 0.274
    XGBoost: 0.264
    LightGBM: 0.256
    Random Forest: 0.206

  Fold 1 Results:
    MAE: 29.8 yards
    RMSE: 41.3 yards
    %Correct: 56.7% (231/408 correct)

Fold 2/5 (Train: 2015-2018, Val: 2019)
  Training XGBoost...     Val MAE: 33.4
  Training LightGBM...    Val MAE: 34.2
  Training Random Forest...Val MAE: 35.1
  Training CatBoost...    Val MAE: 32.9

  Ensemble Weights:
    CatBoost: 0.269
    XGBoost: 0.265
    LightGBM: 0.259
    Random Forest: 0.207

  Fold 2 Results:
    MAE: 31.5 yards
    RMSE: 43.8 yards
    %Correct: 54.2% (221/408 correct)

... (Folds 3-5)

=== Average Across All Folds ===
  Mean MAE: 32.1 ± 1.4 yards
  Mean RMSE: 42.6 ± 1.2 yards
  Mean %Correct: 55.4% ± 1.1%

=== Feature Importances (XGBoost Average) ===
  1. weighted_avg:           35.2%
  2. last_3_avg:             11.8%
  3. opp_def_pass_ypa:        9.5%
  4. success_rate_3wk:        7.3%
  5. career_avg:              6.1%
  6. variance_cv:             4.8%
  7. games_played:            4.2%
  8. last_5_avg:              3.7%
  9. opp_def_pass_td_rate:    2.9%
  10. opponent_BUF:           2.1%
  ... (Top 20 shown)

=== Validation Complete ===
Status: PASS (Mean %Correct > 52% threshold)
Recommendation: Proceed to full model training
```

---

**END OF IMPLEMENTATION PLAN**
