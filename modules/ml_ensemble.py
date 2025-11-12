"""
ML Ensemble Model

5-model diverse ensemble for player props prediction:
- LightGBM (gradient boosted trees)
- CatBoost (gradient boosted trees with native categorical handling)
- ExtraTreesRegressor (extremely randomized trees)
- PyTorch Neural Network (deep learning for complex interactions)
- PLS Regression (supervised dimensionality reduction)

Features:
- sklearn Pipeline for preprocessing
- TimeSeriesSplit cross-validation
- Hyperparameter optimization via RandomizedSearchCV
- Weighted ensemble averaging (based on MAE + MSE)
- Model persistence

Usage:
    ensemble = PropEnsembleModel(prop_type='passing_yards')
    ensemble.train(train_df, optimize_hyperparams=True)
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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, RegressorMixin

import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostRegressor

from modules.logger import get_logger
from modules.constants import CACHE_DIR
from modules.prop_types import get_prop_feature_config, should_filter_features

logger = get_logger(__name__)


class PyTorchRegressor(BaseEstimator, RegressorMixin):
    """
    PyTorch neural network wrapper compatible with sklearn Pipeline.

    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes (default: [128, 64, 32])
        dropout: Dropout rate (default: 0.3)
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size for training (default: 64)
        random_state: Random seed for reproducibility (default: 42)
    """

    def __init__(self, input_dim=35, hidden_layers=[128, 64, 32], dropout=0.3,
                 learning_rate=0.001, epochs=100, batch_size=64, random_state=42):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def _build_network(self):
        """Build the neural network architecture."""
        layers = []
        prev_size = self.input_dim

        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size

        # Output layer (single value for regression)
        layers.append(nn.Linear(prev_size, 1))

        return nn.Sequential(*layers)

    def fit(self, X, y):
        """Train the neural network."""
        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Update input dimension based on actual data
        self.input_dim = X.shape[1]

        # Build model
        self.model = self._build_network()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).view(-1, 1)

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
            predictions = self.model(X_tensor).numpy().flatten()

        return predictions


class PropEnsembleModel:
    """
    5-model diverse ensemble for player prop prediction.

    Combines LightGBM, CatBoost, ExtraTreesRegressor, PyTorch Neural Network,
    and PLS Regression with weighted averaging based on cross-validation
    performance (MAE + MSE).

    The ensemble includes diverse model types:
    - Tree-based (LightGBM, CatBoost, ExtraTrees) for non-linear patterns
    - Neural network (PyTorch) for complex feature interactions
    - Dimensionality reduction (PLS) for supervised feature compression
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

    def _build_extratrees_pipeline(self, scale_features: List[str], nan_features: List[str], categorical_features: List[str]) -> Pipeline:
        """Build ExtraTreesRegressor pipeline with preprocessing."""

        transformers = [
            ('num', StandardScaler(), scale_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ]

        # Add NaN features with passthrough (ExtraTrees handles NaN natively)
        if nan_features:
            transformers.append(('nan', 'passthrough', nan_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        # ExtraTreesRegressor (extremely randomized trees - more diverse than RandomForest)
        model = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline

    def _build_pytorch_pipeline(self, scale_features: List[str], nan_features: List[str], categorical_features: List[str]) -> Pipeline:
        """Build PyTorch neural network pipeline with preprocessing."""

        transformers = [
            ('num', StandardScaler(), scale_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ]

        # PyTorch doesn't handle NaN - use SimpleImputer (median strategy)
        if nan_features:
            transformers.append(('nan', SimpleImputer(strategy='median'), nan_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        # PyTorch neural network (deep learning for complex feature interactions)
        model = PyTorchRegressor(
            hidden_layers=[128, 64, 32],
            dropout=0.3,
            learning_rate=0.001,
            epochs=100,
            batch_size=64,
            random_state=42
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline

    def _build_pls_pipeline(self, scale_features: List[str], nan_features: List[str], categorical_features: List[str]) -> Pipeline:
        """Build PLS Regression pipeline with preprocessing."""

        transformers = [
            ('num', StandardScaler(), scale_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ]

        # PLS doesn't handle NaN - use SimpleImputer (median strategy)
        if nan_features:
            transformers.append(('nan', SimpleImputer(strategy='median'), nan_features))

        preprocessor = ColumnTransformer(transformers=transformers)

        # PLS Regression (supervised dimensionality reduction)
        model = PLSRegression(
            n_components=10,  # Will be tuned
            scale=False  # Already scaled in preprocessing
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return pipeline

    def _get_hyperparam_distributions(self, model_name: str) -> Dict:
        """
        Get hyperparameter distributions for RandomizedSearchCV.

        Args:
            model_name: Name of model ('lightgbm', 'catboost', 'extratrees', 'pytorch', 'pls')

        Returns:
            Dictionary of hyperparameter distributions
        """
        distributions = {
            'lightgbm': {
                'model__n_estimators': [300, 500],
                'model__max_depth': [6, 8, 10],
                'model__learning_rate': [0.03, 0.05, 0.1],
                'model__num_leaves': [31, 63],
                'model__subsample': [0.8],
                'model__colsample_bytree': [0.8]
            },
            'catboost': {
                'model__iterations': [300, 500],
                'model__depth': [6, 8, 10],
                'model__learning_rate': [0.03, 0.05, 0.1]
            },
            'extratrees': {
                'model__n_estimators': [200, 300, 500],
                'model__max_depth': [6, 8, 10, None],
                'model__min_samples_split': [2, 5, 10],
                'model__max_features': ['sqrt', 'log2', None]
            },
            'pytorch': {
                'model__hidden_layers': [[128, 64, 32], [256, 128, 64], [64, 32, 16]],
                'model__dropout': [0.2, 0.3, 0.4],
                'model__learning_rate': [0.001, 0.0005, 0.01],
                'model__epochs': [50, 100, 150]
            },
            'pls': {
                'model__n_components': [5, 10, 15, 20]
            }
        }

        return distributions.get(model_name, {})

    def train(
        self,
        train_df: pl.DataFrame,
        n_splits: int = 5,
        verbose: bool = True,
        optimize_hyperparams: bool = False,
        n_iter: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Train ensemble with cross-validation.

        Args:
            train_df: Training data with 'target' column
            n_splits: Number of time series splits for CV
            verbose: Print progress
            optimize_hyperparams: If True, run RandomizedSearchCV for hyperparameter tuning
            n_iter: Number of iterations for RandomizedSearchCV (ignored if optimize_hyperparams=False)

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

        # CRITICAL: Sort by temporal order for proper TimeSeriesSplit
        # Without sorting, TimeSeriesSplit will create random (not temporal) splits
        # since multi-year data may be shuffled
        if 'season' in train_pd_filtered.columns and 'week' in train_pd_filtered.columns:
            train_pd_filtered = train_pd_filtered.sort_values(['season', 'week', 'player_id']).reset_index(drop=True)
            logger.info("Sorted training data by (season, week, player_id) for temporal cross-validation")

        X = train_pd_filtered[self.feature_columns]
        y = train_pd_filtered['target']

        # Build pipelines with NaN handling
        pipelines = {
            'lightgbm': self._build_lightgbm_pipeline(scale_features, nan_features, categorical_features),
            'catboost': self._build_catboost_pipeline(scale_features, nan_features, categorical_features),
            'extratrees': self._build_extratrees_pipeline(scale_features, nan_features, categorical_features),
            'pytorch': self._build_pytorch_pipeline(scale_features, nan_features, categorical_features),
            'pls': self._build_pls_pipeline(scale_features, nan_features, categorical_features)
        }

        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            logger.info(f"\n{'='*60}")
            logger.info("Hyperparameter Optimization (RandomizedSearchCV)")
            logger.info(f"{'='*60}")
            logger.info(f"Iterations per model: {n_iter}")
            logger.info(f"CV splits: {n_splits}")

            tscv_opt = TimeSeriesSplit(n_splits=n_splits)
            optimized_pipelines = {}

            for model_name, pipeline in pipelines.items():
                logger.info(f"\n  Optimizing {model_name}...")

                param_distributions = self._get_hyperparam_distributions(model_name)

                if not param_distributions:
                    logger.info(f"    No hyperparameters to tune for {model_name}, using defaults")
                    optimized_pipelines[model_name] = pipeline
                    continue

                # Run randomized search
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_distributions,
                    n_iter=n_iter,
                    cv=tscv_opt,
                    scoring='neg_mean_absolute_error',
                    n_jobs=2,  # Limit to 2 cores to avoid system freeze
                    random_state=42,
                    verbose=0
                )

                search.fit(X, y)

                logger.info(f"    Best MAE: {-search.best_score_:.2f}")
                logger.info(f"    Best params: {search.best_params_}")

                # Use best estimator
                optimized_pipelines[model_name] = search.best_estimator_

            pipelines = optimized_pipelines
            logger.info(f"\n{'='*60}")
            logger.info("Hyperparameter optimization complete")
            logger.info(f"{'='*60}")

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
        mse_scores = []

        for model_name in pipelines.keys():
            mae_mean = np.mean(cv_results[model_name]['mae'])
            mae_std = np.std(cv_results[model_name]['mae'])
            mse_mean = np.mean(cv_results[model_name]['mse'])
            mse_std = np.std(cv_results[model_name]['mse'])
            rmse_mean = np.mean(cv_results[model_name]['rmse'])
            rmse_std = np.std(cv_results[model_name]['rmse'])

            performance_summary[model_name] = {
                'mae_mean': mae_mean,
                'mae_std': mae_std,
                'mse_mean': mse_mean,
                'mse_std': mse_std,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std
            }

            mae_scores.append(mae_mean)
            mse_scores.append(mse_mean)

            logger.info(f"{model_name:15s}: MAE={mae_mean:.2f}±{mae_std:.2f}, MSE={mse_mean:.2f}±{mse_std:.2f}, RMSE={rmse_mean:.2f}±{rmse_std:.2f}")

        # Calculate weights (inverse MAE + inverse MSE, normalized)
        # Using both MAE and MSE to penalize both average errors and catastrophic predictions
        mae_scores = np.array(mae_scores)
        mse_scores = np.array(mse_scores)

        # Normalize scores to 0-1 range before inverting (so both metrics have similar scale)
        mae_normalized = (mae_scores - mae_scores.min()) / (mae_scores.max() - mae_scores.min() + 1e-6)
        mse_normalized = (mse_scores - mse_scores.min()) / (mse_scores.max() - mse_scores.min() + 1e-6)

        # Combined score (lower is better)
        combined_score = 0.5 * mae_normalized + 0.5 * mse_normalized

        # Invert (so higher is better) and normalize to weights
        inverse_combined = 1.0 / (combined_score + 1e-6)
        weights = inverse_combined / inverse_combined.sum()

        for (model_name, weight) in zip(pipelines.keys(), weights):
            self.model_weights[model_name] = weight

        logger.info(f"\n{'='*60}")
        logger.info("Ensemble Weights (based on MAE + MSE):")
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

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance from trained ensemble.

        Aggregates importance across all 5 models weighted by their ensemble weights.
        Returns sorted dictionary of {feature_name: importance_score}.

        Note: Only tree models (LightGBM, CatBoost, ExtraTrees) provide feature importance.
        PyTorch and PLS models don't have native feature importance.

        Returns:
            Dictionary mapping feature names to importance scores (0-1, normalized)
        """
        if not self.models:
            raise ValueError("Model not trained. Call train() first.")

        # Initialize importance dictionary
        feature_importance = {feat: 0.0 for feat in self.feature_columns}

        # Extract importance from each model
        for model_name, pipeline in self.models.items():
            model_weight = self.model_weights[model_name]

            # Get the actual model from the pipeline (last step)
            model = pipeline.named_steps['model']

            # Extract feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree models (XGBoost, LightGBM, CatBoost, RandomForest)
                importances = model.feature_importances_
            elif hasattr(model, 'get_booster'):
                # XGBoost alternative method
                booster = model.get_booster()
                importance_dict = booster.get_score(importance_type='weight')
                # Map to feature columns
                importances = np.array([importance_dict.get(f'f{i}', 0.0) for i in range(len(self.feature_columns))])
            else:
                # Model doesn't support feature importance
                logger.warning(f"{model_name} doesn't support feature importance extraction")
                continue

            # Normalize importance to sum to 1
            if importances.sum() > 0:
                importances = importances / importances.sum()

            # Add weighted importance to total
            for i, feat in enumerate(self.feature_columns):
                feature_importance[feat] += importances[i] * model_weight

        # Sort by importance (descending)
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        return sorted_importance

    def print_feature_importance(self, top_n: int = 20):
        """
        Print top N most important features.

        Args:
            top_n: Number of top features to display (default 20)
        """
        importance = self.get_feature_importance()

        print(f"\n{'='*60}")
        print(f"Top {top_n} Most Important Features")
        print(f"{'='*60}")
        print(f"{'Feature':<35} {'Importance':>10}")
        print(f"{'-'*60}")

        for i, (feat, imp) in enumerate(list(importance.items())[:top_n]):
            print(f"{feat:<35} {imp:>10.4f}")

        print(f"{'='*60}\n")

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
