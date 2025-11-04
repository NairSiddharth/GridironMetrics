"""
Train ML Ensembles for All Prop Types

Generates training data and trains ensemble models for all 7 prop types:
- passing_yards (QBs)
- passing_tds (QBs)
- rushing_yards (RBs)
- rushing_tds (RBs)
- receptions (WR/TE)
- receiving_yards (WR/TE)
- receiving_tds (WR/TE)

Usage:
    # Train all prop types
    python scripts/train_all_prop_types.py

    # Train specific prop types
    python scripts/train_all_prop_types.py --props passing_tds rushing_yards
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.ml_training_data_builder import TrainingDataBuilder
from modules.ml_ensemble import PropEnsembleModel
from modules.constants import CACHE_DIR
import polars as pl
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# All available prop types
ALL_PROP_TYPES = [
    'passing_yards',
    'passing_tds',
    'rushing_yards',
    'rushing_tds',
    'receptions',
    'receiving_yards',
    'receiving_tds'
]


def train_prop_type(prop_type: str, start_year: int = 2015, end_year: int = 2024, force_rebuild: bool = False):
    """
    Train ensemble for a specific prop type.

    Args:
        prop_type: Prop type to train
        start_year: First year for training data
        end_year: Last year for training data
        force_rebuild: Force rebuild of training data even if exists
    """
    print(f"\n{'='*80}")
    print(f"TRAINING: {prop_type.upper()}")
    print(f"{'='*80}")

    # Check if training data exists
    training_data_path = Path(CACHE_DIR) / "ml_training_data" / f"{prop_type}_{start_year}_{end_year}.parquet"

    if training_data_path.exists() and not force_rebuild:
        print(f"\nLoading existing training data from {training_data_path}")
        train_df = pl.read_parquet(training_data_path)
        print(f"Loaded {len(train_df):,} examples")
    else:
        print(f"\nGenerating training data for {prop_type}...")
        print(f"Years: {start_year}-{end_year}")

        builder = TrainingDataBuilder()
        train_df = builder.build_training_dataset(
            start_year=start_year,
            end_year=end_year,
            prop_type=prop_type
        )

        if len(train_df) == 0:
            print(f"ERROR: No training data generated for {prop_type}")
            return None

    # Train ensemble
    print(f"\nTraining ensemble for {prop_type}...")

    ensemble = PropEnsembleModel(prop_type=prop_type)
    performance = ensemble.train(train_df, n_splits=5, verbose=False)

    # Save model
    ensemble.save()

    # Print summary
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE: {prop_type.upper()}")
    print(f"{'='*80}")
    print(f"Training examples: {len(train_df):,}")
    print(f"Years: {start_year}-{end_year}")

    print(f"\nCross-Validation Results:")
    for model_name, metrics in performance.items():
        print(f"  {model_name:15s}: MAE={metrics['mae_mean']:.2f}±{metrics['mae_std']:.2f}")

    best_model = min(performance.items(), key=lambda x: x[1]['mae_mean'])
    print(f"\nBest model: {best_model[0]} (MAE: {best_model[1]['mae_mean']:.2f})")

    model_path = Path(CACHE_DIR) / "ml_models" / prop_type / f"{prop_type}_ensemble.joblib"
    print(f"Model saved: {model_path}")

    return {
        'prop_type': prop_type,
        'examples': len(train_df),
        'performance': performance,
        'model_path': model_path
    }


def main():
    parser = argparse.ArgumentParser(description='Train ML ensembles for prop types')
    parser.add_argument('--props', nargs='+', choices=ALL_PROP_TYPES, default=None,
                        help='Specific prop types to train (default: all except passing_yards)')
    parser.add_argument('--start-year', type=int, default=2015,
                        help='First year for training data (default: 2015)')
    parser.add_argument('--end-year', type=int, default=2024,
                        help='Last year for training data (default: 2024)')
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Force rebuild of training data even if exists')

    args = parser.parse_args()

    # Determine which prop types to train
    if args.props:
        prop_types_to_train = args.props
    else:
        # Default: train all except passing_yards (already done)
        prop_types_to_train = [p for p in ALL_PROP_TYPES if p != 'passing_yards']

    print(f"\n{'='*80}")
    print("ML ENSEMBLE TRAINING - MULTIPLE PROP TYPES")
    print(f"{'='*80}")
    print(f"\nProp types to train: {', '.join(prop_types_to_train)}")
    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"Force rebuild: {args.force_rebuild}")

    results = {}

    for prop_type in prop_types_to_train:
        try:
            result = train_prop_type(
                prop_type=prop_type,
                start_year=args.start_year,
                end_year=args.end_year,
                force_rebuild=args.force_rebuild
            )
            if result:
                results[prop_type] = result
        except Exception as e:
            print(f"\nERROR training {prop_type}: {e}")
            import traceback
            traceback.print_exc()
            results[prop_type] = {'error': str(e)}

    # Print final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Prop Type':<20} {'Examples':<12} {'Best MAE':<12} {'Status':<12}")
    print("-" * 80)

    for prop_type in prop_types_to_train:
        if prop_type in results:
            result = results[prop_type]
            if 'error' in result:
                print(f"{prop_type:<20} {'N/A':<12} {'N/A':<12} {'FAILED':<12}")
            else:
                perf = result['performance']
                best_mae = min([m['mae_mean'] for m in perf.values()])
                print(f"{prop_type:<20} {result['examples']:<12,} {best_mae:<12.2f} {'SUCCESS':<12}")

    print(f"\n{'='*80}")
    print(f"Training complete for {len([r for r in results.values() if 'error' not in r])} prop types")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    results = main()
