"""
Generate Full Training Dataset (2015-2023)

Creates training data for passing_yards using all available historical data.
This will take approximately 30-40 minutes to process all years.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.ml_training_data_builder import TrainingDataBuilder
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("\n" + "="*60)
print("Full Training Data Generation (2015-2023)")
print("="*60)
print("\nGenerating passing_yards training data...")
print("This will take approximately 30-40 minutes")
print("="*60 + "\n")

builder = TrainingDataBuilder()

# Build passing_yards dataset for full historical range
train_df = builder.build_training_dataset(
    start_year=2015,
    end_year=2023,
    prop_type='passing_yards'
)

if len(train_df) > 0:
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total examples: {len(train_df):,}")
    print(f"Features: {len(train_df.columns) - 3}")  # Exclude target, player_id, year
    print(f"Years: 2015-2023")

    print("\n" + "="*60)
    print("TARGET STATISTICS")
    print("="*60)
    print(f"Mean: {train_df['target'].mean():.2f}")
    print(f"Std: {train_df['target'].std():.2f}")
    print(f"Min: {train_df['target'].min():.2f}")
    print(f"Max: {train_df['target'].max():.2f}")

    # Year distribution
    print("\n" + "="*60)
    print("EXAMPLES PER YEAR")
    print("="*60)
    year_counts = train_df.group_by('year').count().sort('year')
    for row in year_counts.iter_rows(named=True):
        print(f"  {row['year']}: {row['count']:,} examples")

print("\n" + "="*60)
print("Full training data ready for model training")
print("="*60)
