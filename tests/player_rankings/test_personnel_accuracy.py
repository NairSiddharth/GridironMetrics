"""
Test script to compare personnel inference accuracy vs. actual participation data.
"""
import nflreadpy as nfl
import polars as pl
from modules.personnel_inference import PersonnelInference

# Load play-by-play and participation data
print("Loading data...")
pbp = nfl.load_pbp([2024])
participation = nfl.load_participation([2024])

# Parse actual personnel from participation
participation = participation.with_columns([
    pl.col('offense_personnel').str.extract(r'(\d+)\s+RB', 1).cast(pl.Int32, strict=False).fill_null(0).alias('rb_count'),
    pl.col('offense_personnel').str.extract(r'(\d+)\s+TE', 1).cast(pl.Int32, strict=False).fill_null(0).alias('te_count')
])

participation = participation.with_columns([
    (pl.col('rb_count').cast(pl.Utf8) + pl.col('te_count').cast(pl.Utf8)).alias('actual_personnel')
])

# Join with PBP data
merged = pbp.join(
    participation.select(['nflverse_game_id', 'play_id', 'actual_personnel']),
    left_on=['game_id', 'play_id'],
    right_on=['nflverse_game_id', 'play_id'],
    how='inner'
)

# Filter to regular offensive plays
merged = merged.filter(
    (pl.col('play_type').is_in(['pass', 'run'])) &
    (pl.col('actual_personnel').is_not_null()) &
    (pl.col('actual_personnel').is_in(['10', '11', '12', '13', '21', '22']))  # Common personnel only
)

print(f"\nAnalyzing {len(merged)} plays...")

# Initialize personnel inference
personnel_inf = PersonnelInference()

# Infer personnel for each play
inferred_personnel = []
for row in merged.iter_rows(named=True):
    play_type = row['play_type']
    down = row['down'] or 1
    ydstogo = row['ydstogo'] or 10
    yardline_100 = row['yardline_100'] or 50
    score_diff = row['score_differential'] or 0
    game_seconds = row['game_seconds_remaining'] or 1800
    
    receiver_pos = None
    air_yards = None
    if play_type == 'pass':
        if row['receiver_player_name']:
            # Try to infer receiver position from name patterns (simplified)
            receiver_pos = 'WR'  # Default
        air_yards = row['air_yards'] or 0
    
    personnel, confidence = personnel_inf.infer_personnel(
        play_type, down, ydstogo, yardline_100, score_diff, game_seconds,
        receiver_pos, air_yards
    )
    inferred_personnel.append(personnel)

merged = merged.with_columns([
    pl.Series('inferred_personnel', inferred_personnel)
])

# Calculate accuracy
total = len(merged)
correct = len(merged.filter(pl.col('actual_personnel') == pl.col('inferred_personnel')))
accuracy = correct / total * 100

print(f"\n{'='*60}")
print(f"PERSONNEL INFERENCE ACCURACY")
print(f"{'='*60}")
print(f"Total plays analyzed: {total:,}")
print(f"Correct predictions: {correct:,}")
print(f"Accuracy: {accuracy:.1f}%")
print(f"{'='*60}\n")

# Show confusion matrix
print("Confusion Matrix (Actual vs. Inferred):")
confusion = merged.group_by(['actual_personnel', 'inferred_personnel']).agg(pl.len().alias('count')).sort(['actual_personnel', 'count'], descending=[False, True])
print(confusion)

# Show accuracy by personnel type
print("\nAccuracy by Personnel Type:")
by_personnel = merged.with_columns([
    (pl.col('actual_personnel') == pl.col('inferred_personnel')).alias('correct')
]).group_by('actual_personnel').agg([
    pl.len().alias('total'),
    pl.col('correct').sum().alias('correct_count')
]).with_columns([
    (pl.col('correct_count') / pl.col('total') * 100).alias('accuracy_pct')
]).sort('total', descending=True)
print(by_personnel)

# Show most common misclassifications
print("\nTop 10 Misclassifications:")
misclassified = merged.filter(pl.col('actual_personnel') != pl.col('inferred_personnel'))
top_errors = misclassified.group_by(['actual_personnel', 'inferred_personnel']).agg(pl.len().alias('count')).sort('count', descending=True).head(10)
print(top_errors)
