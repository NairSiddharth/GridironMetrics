"""
ML Ensemble Model

4-model ensemble for player props prediction:
- XGBoost
- LightGBM
- CatBoost (native categorical handling)
- Random Forest

Features:
- sklearn Pipeline for preprocessing
- TimeSeriesSplit cross-validation
- Weighted ensemble averaging
- Model persistence

Usage:
    ensemble = PropEnsembleModel(prop_type='passing_yards')
    ensemble.train(train_df)
    prediction = ensemble.predict(features)
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from modules.logger import get_logger
from modules.constants import CACHE_DIR
from modules.prop_types import get_prop_feature_config, should_filter_features

logger = get_logger(__name__)


class PropEnsembleModel:
    """
    4-model ensemble for player prop prediction.

    Combines XGBoost, LightGBM, CatBoost, and Random Forest with
    weighted averaging based on cross-validation performance.
    """

    def __init__(self, prop_type: str = 'passing_yards'):
        """
        Initialize ensemble model.

        Args:
            prop_type: Type of prop to predict (e.g., 'passing_yards')
        """
        self.prop_type = prop_type
        self.models = {}
        self.model_weights = {}
        self.feature_columns = None
        self.categorical_features = ['opponent', 'position']

        # Model save directory
        self.model_dir = Path(CACHE_DIR) / "ml_models" / prop_type
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _get_feature_columns(self, df: pl.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Separate numeric, NaN-friendly, and categorical features.

        Args:
            df: Training dataframe

        Returns:
            (scale_features, nan_features, categorical_features)
        """
        # Exclude metadata columns
        exclude_cols = {'target', 'player_id', 'year'}
        all_features = [col for col in df.columns if col not in exclude_cols]

        # NaN-friendly features (tree models handle NaN natively, no scaling needed)
        # These features may have NaN for missing data (pre-2016 NextGen, missing weather, etc.)
        nan_friendly = {
            'avg_separation', 'avg_cushion',  # NextGen (pre-2016)
            'game_temp', 'game_wind',  # Weather (if missing)
            'is_home', 'is_dome', 'division_game',  # Game context (if missing)
            'vegas_total', 'vegas_spread'  # Vegas lines (if missing)
        }

        # Identify categorical features
        categorical_features = [col for col in all_features if col in self.categorical_features]

        # NaN features that exist in this dataset
        nan_features = [col for col in all_features if col in nan_friendly]

        # Features to scale (exclude categorical and NaN-friendly)
        scale_features = [
            col for col in all_features
            if col not in categorical_features and col not in nan_friendly
        ]

        return scale_features, nan_features, categorical_features

    def _build_xgboost_pipeline(self, scale_features: List[str], nan_features: List[str], categorical_features: List[str]) -> Pipeline:
        """Build XGBoost pipeline with preprocessing."""

        # Preprocessing: scale numeric, pass through NaN features, encode categorical
        transformers = [
            ('num', StandardScaler(), scale_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ]

        # Add NaN features with passthrough (XGBoost handles NaN natively)
        if nan_features:
            transformers.append(('nan', 'passthrough', nan_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        # XGBoost model (tree_method supports NaN)
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # Supports NaN natively
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline

    def _build_lightgbm_pipeline(self, scale_features: List[str], nan_features: List[str], categorical_features: List[str]) -> Pipeline:
        """Build LightGBM pipeline with preprocessing."""

        transformers = [
            ('num', StandardScaler(), scale_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ]

        # Add NaN features with passthrough (LightGBM handles NaN natively)
        if nan_features:
            transformers.append(('nan', 'passthrough', nan_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        # LightGBM model (handles NaN natively)
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline

    def _build_catboost_pipeline(self, scale_features: List[str], nan_features: List[str], categorical_features: List[str]) -> Pipeline:
        """Build CatBoost pipeline (native categorical handling)."""

        # CatBoost handles categoricals and NaN natively, so we only scale numerics
        transformers = [
            ('num', StandardScaler(), scale_features),
            ('cat', 'passthrough', categorical_features)  # Keep categorical as-is
        ]

        # Add NaN features with passthrough (CatBoost handles NaN natively)
        if nan_features:
            transformers.append(('nan', 'passthrough', nan_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        # Get categorical feature indices (after preprocessing)
        cat_feature_indices = list(range(len(scale_features), len(scale_features) + len(categorical_features)))

        # CatBoost model (handles NaN natively)
        model = CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=False,
            cat_features=cat_feature_indices
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline

    def _build_random_forest_pipeline(self, scale_features: List[str], nan_features: List[str], categorical_features: List[str]) -> Pipeline:
        """Build Random Forest pipeline with preprocessing."""

        transformers = [
            ('num', StandardScaler(), scale_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ]

        # Random Forest doesn't handle NaN natively - use SimpleImputer (median strategy)
        if nan_features:
            transformers.append(('nan', SimpleImputer(strategy='median'), nan_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        # Random Forest model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline

    def train(
        self,
        train_df: pl.DataFrame,
        n_splits: int = 5,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Train ensemble with cross-validation.

        Args:
            train_df: Training data with 'target' column
            n_splits: Number of time series splits for CV
            verbose: Print progress

        Returns:
            Dictionary of model performance metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Ensemble: {self.prop_type}")
        logger.info(f"{'='*60}")
        logger.info(f"Training examples: {len(train_df):,}")
        logger.info(f"CV splits: {n_splits}")

        # Convert to pandas for sklearn compatibility
        train_pd = train_df.to_pandas()

        # Apply feature filtering if configured for this prop type
        if should_filter_features(self.prop_type):
            config = get_prop_feature_config(self.prop_type)
            allowed_features = config['include']

            # Filter to only allowed features (exclude metadata)
            exclude_cols = {'target', 'player_id', 'year'}
            all_available = [col for col in train_pd.columns if col not in exclude_cols]

            # Keep only features that are both allowed and available
            filtered_features = [col for col in all_available if col in allowed_features]

            logger.info(f"Feature filtering ENABLED for {self.prop_type}")
            logger.info(f"  Available features: {len(all_available)}")
            logger.info(f"  Allowed features: {len(allowed_features)}")
            logger.info(f"  Features after filtering: {len(filtered_features)}")

            # Create filtered dataframe
            filtered_df = train_pd[filtered_features + ['target', 'player_id', 'year']]
            train_df_filtered = pl.from_pandas(filtered_df)
        else:
            logger.info(f"Feature filtering DISABLED for {self.prop_type} (using all features)")
            train_df_filtered = train_df

        # Separate features into scale, NaN-friendly, and categorical
        scale_features, nan_features, categorical_features = self._get_feature_columns(train_df_filtered)
        self.feature_columns = scale_features + nan_features + categorical_features

        logger.info(f"Features to scale: {len(scale_features)}")
        logger.info(f"NaN-friendly features: {len(nan_features)}")
        logger.info(f"Categorical features: {len(categorical_features)}")
        logger.info(f"Total features in model: {len(self.feature_columns)}")

        # Get X and y from filtered data
        train_pd_filtered = train_df_filtered.to_pandas()
        X = train_pd_filtered[self.feature_columns]
        y = train_pd_filtered['target']

        # Build pipelines with NaN handling
        pipelines = {
            'xgboost': self._build_xgboost_pipeline(scale_features, nan_features, categorical_features),
            'lightgbm': self._build_lightgbm_pipeline(scale_features, nan_features, categorical_features),
            'catboost': self._build_catboost_pipeline(scale_features, nan_features, categorical_features),
            'random_forest': self._build_random_forest_pipeline(scale_features, nan_features, categorical_features)
        }

        # Cross-validation with TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {name: {'mae': [], 'mse': [], 'rmse': []} for name in pipelines.keys()}

        logger.info(f"\nCross-Validation:")

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"\n  Fold {fold_idx}/{n_splits}:")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            logger.info(f"    Train: {len(X_train_fold)}, Val: {len(X_val_fold)}")

            for model_name, pipeline in pipelines.items():
                # Train
                pipeline.fit(X_train_fold, y_train_fold)

                # Predict
                y_pred = pipeline.predict(X_val_fold)

                # Metrics
                mae = mean_absolute_error(y_val_fold, y_pred)
                mse = mean_squared_error(y_val_fold, y_pred)
                rmse = np.sqrt(mse)

                cv_results[model_name]['mae'].append(mae)
                cv_results[model_name]['mse'].append(mse)
                cv_results[model_name]['rmse'].append(rmse)

                if verbose:
                    logger.info(f"    {model_name:15s}: MAE={mae:.2f}, RMSE={rmse:.2f}")

        # Calculate mean performance and weights
        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Results (Mean ± Std):")
        logger.info(f"{'='*60}")

        performance_summary = {}
        mae_scores = []

        for model_name in pipelines.keys():
            mae_mean = np.mean(cv_results[model_name]['mae'])
            mae_std = np.std(cv_results[model_name]['mae'])
            rmse_mean = np.mean(cv_results[model_name]['rmse'])
            rmse_std = np.std(cv_results[model_name]['rmse'])

            performance_summary[model_name] = {
                'mae_mean': mae_mean,
                'mae_std': mae_std,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std
            }

            mae_scores.append(mae_mean)

            logger.info(f"{model_name:15s}: MAE={mae_mean:.2f}±{mae_std:.2f}, RMSE={rmse_mean:.2f}±{rmse_std:.2f}")

        # Calculate weights (inverse MAE, normalized)
        mae_scores = np.array(mae_scores)
        inverse_mae = 1.0 / (mae_scores + 1e-6)  # Avoid division by zero
        weights = inverse_mae / inverse_mae.sum()

        for (model_name, weight) in zip(pipelines.keys(), weights):
            self.model_weights[model_name] = weight

        logger.info(f"\n{'='*60}")
        logger.info("Ensemble Weights (based on inverse MAE):")
        logger.info(f"{'='*60}")
        for model_name, weight in self.model_weights.items():
            logger.info(f"{model_name:15s}: {weight:.4f}")

        # Train final models on full dataset
        logger.info(f"\n{'='*60}")
        logger.info("Training final models on full dataset...")
        logger.info(f"{'='*60}")

        for model_name, pipeline in pipelines.items():
            logger.info(f"  Training {model_name}...")
            pipeline.fit(X, y)
            self.models[model_name] = pipeline

        logger.info(f"\nEnsemble training complete!")

        return performance_summary

    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict using weighted ensemble.

        Args:
            features: Dictionary of feature values

        Returns:
            Weighted ensemble prediction
        """
        if not self.models:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to DataFrame
        import pandas as pd
        features_df = pd.DataFrame([features])[self.feature_columns]

        # Get predictions from each model
        predictions = []
        weights = []

        for model_name, pipeline in self.models.items():
            pred = pipeline.predict(features_df)[0]
            weight = self.model_weights[model_name]

            predictions.append(pred)
            weights.append(weight)

        # Weighted average
        weighted_pred = np.average(predictions, weights=weights)

        return weighted_pred

    def predict_batch(self, features_df: pl.DataFrame) -> np.ndarray:
        """
        Predict for multiple examples.

        Args:
            features_df: Polars DataFrame with feature columns

        Returns:
            Array of predictions
        """
        if not self.models:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to pandas
        features_pd = features_df.to_pandas()[self.feature_columns]

        # Get predictions from each model
        all_predictions = []
        weights = []

        for model_name, pipeline in self.models.items():
            preds = pipeline.predict(features_pd)
            weight = self.model_weights[model_name]

            all_predictions.append(preds)
            weights.append(weight)

        # Weighted average across models
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_examples)
        weights = np.array(weights)

        weighted_preds = np.average(all_predictions, axis=0, weights=weights)

        return weighted_preds

    def save(self, filename: Optional[str] = None):
        """
        Save ensemble to disk.

        Args:
            filename: Optional filename (default: {prop_type}_ensemble.joblib)
        """
        if filename is None:
            filename = f"{self.prop_type}_ensemble.joblib"

        filepath = self.model_dir / filename

        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features,
            'prop_type': self.prop_type
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Ensemble saved to: {filepath}")

    def load(self, filename: Optional[str] = None):
        """
        Load ensemble from disk.

        Args:
            filename: Optional filename (default: {prop_type}_ensemble.joblib)
        """
        if filename is None:
            filename = f"{self.prop_type}_ensemble.joblib"

        filepath = self.model_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.models = model_data['models']
        self.model_weights = model_data['model_weights']
        self.feature_columns = model_data['feature_columns']
        self.categorical_features = model_data['categorical_features']
        self.prop_type = model_data['prop_type']

        logger.info(f"Ensemble loaded from: {filepath}")


if __name__ == "__main__":
    # Test ensemble training on 2023 passing_yards data
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*60)
    print("ML Ensemble - Test")
    print("="*60)
    print("\nTesting ensemble training on 2023 passing_yards data")
    print("="*60 + "\n")

    # Load training data
    training_data_path = Path(CACHE_DIR) / "ml_training_data" / "passing_yards_2023_2023.parquet"

    if not training_data_path.exists():
        print(f"ERROR: Training data not found at {training_data_path}")
        print("Please run ml_training_data_builder.py first")
        exit(1)

    train_df = pl.read_parquet(training_data_path)
    print(f"Loaded {len(train_df):,} training examples")

    # Initialize and train ensemble
    ensemble = PropEnsembleModel(prop_type='passing_yards')
    performance = ensemble.train(train_df, n_splits=3, verbose=True)

    # Save model
    ensemble.save()

    # Test prediction on first example
    print("\n" + "="*60)
    print("Test Prediction")
    print("="*60)

    test_example = train_df.row(0, named=True)
    features = {k: v for k, v in test_example.items() if k not in {'target', 'player_id', 'year'}}
    actual = test_example['target']

    prediction = ensemble.predict(features)

    print(f"Actual: {actual:.2f} yards")
    print(f"Predicted: {prediction:.2f} yards")
    print(f"Error: {abs(prediction - actual):.2f} yards")

    print("\n" + "="*60)
    print("Ensemble ready for production use")
    print("="*60)
