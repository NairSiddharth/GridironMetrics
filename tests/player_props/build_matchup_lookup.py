"""
Build Player vs Team Matchup Lookup Table

Pre-aggregates historical matchup stats for all WR-team combinations.
Creates a lookup table to eliminate expensive PBP loading during feature engineering.

Output: cache/ml_training_data/matchup_lookup.parquet

Columns:
- player_id: GSIS ID
- opponent: Team code
- season: Year
- week: Week number
- wr_vs_team_avg_yards_last3: Avg receiving yards in last 3 career meetings
- wr_vs_team_receptions_last3: Avg receptions in last 3 career meetings

One-time computation: ~30-45 minutes
Reusable for all future dataset rebuilds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
from modules.constants import CACHE_DIR
from modules.logger import get_logger

logger = get_logger(__name__)


def main():
    print("="*80)
    print("BUILDING PLAYER VS TEAM MATCHUP LOOKUP TABLE")
    print("="*80)

    # Step 1: Load and filter PBP data year by year to avoid schema mismatches
    print("\nStep 1: Loading PBP data (2009-2024) and extracting WR plays...")
    wr_plays_list = []

    for year in range(2009, 2025):
        pbp_file = Path(CACHE_DIR) / "pbp" / f"pbp_{year}.parquet"
        if pbp_file.exists():
            try:
                year_pbp = pl.read_parquet(pbp_file)

                # Filter to WR receiving plays immediately
                year_wr_plays = year_pbp.filter(
                    (pl.col('receiver_player_id').is_not_null()) &
                    (pl.col('defteam').is_not_null())
                ).select([
                    'receiver_player_id',
                    'defteam',
                    'season',
                    'week',
                    'receiving_yards',
                    'complete_pass'
                ])

                wr_plays_list.append(year_wr_plays)
                print(f"  {year}: {len(year_wr_plays):,} WR plays")
            except Exception as e:
                print(f"  Error loading {year}: {e}")

    if len(wr_plays_list) == 0:
        print("ERROR: No PBP data found!")
        return

    # Concatenate filtered data (smaller schema, less likely to have mismatches)
    print(f"\nConcatenating {len(wr_plays_list)} years of WR plays...")
    wr_plays = pl.concat(wr_plays_list, how="diagonal")
    print(f"Total WR receiving plays: {len(wr_plays):,}")

    # Step 2: Get all unique (player_id, opponent, season, week) combinations
    # These are all the game-level matchups we need to compute
    print("\nStep 2: Identifying unique matchups...")
    unique_matchups = wr_plays.select([
        'receiver_player_id',
        'defteam',
        'season',
        'week'
    ]).unique().sort(['receiver_player_id', 'defteam', 'season', 'week'])

    print(f"Unique player-opponent-week matchups: {len(unique_matchups):,}")

    # Step 3: For each matchup, calculate historical stats
    print("\nStep 3: Calculating historical matchup stats...")
    print("This will take 30-45 minutes...")

    lookup_rows = []
    total_matchups = len(unique_matchups)

    for idx, row in enumerate(unique_matchups.iter_rows(named=True)):
        player_id = row['receiver_player_id']
        opponent = row['defteam']
        season = row['season']
        week = row['week']

        # Progress indicator
        if (idx + 1) % 1000 == 0:
            pct = ((idx + 1) / total_matchups) * 100
            print(f"  Progress: {idx + 1:,} / {total_matchups:,} ({pct:.1f}%)")

        # Find all prior games where this player faced this opponent
        historical_plays = wr_plays.filter(
            (pl.col('receiver_player_id') == player_id) &
            (pl.col('defteam') == opponent) &
            (
                (pl.col('season') < season) |  # Prior seasons
                ((pl.col('season') == season) & (pl.col('week') < week))  # Or earlier weeks this season
            )
        )

        if len(historical_plays) == 0:
            # No historical matchups
            lookup_rows.append({
                'player_id': player_id,
                'opponent': opponent,
                'season': season,
                'week': week,
                'wr_vs_team_avg_yards_last3': None,  # Will be NaN in parquet
                'wr_vs_team_receptions_last3': None
            })
            continue

        # Group by game to get per-game stats
        game_stats = historical_plays.group_by(['season', 'week']).agg([
            pl.col('receiving_yards').sum().alias('yards'),
            pl.col('complete_pass').sum().alias('receptions')
        ]).sort(['season', 'week'], descending=True)

        # Take last 3 games (most recent)
        last_3_games = game_stats.head(3)

        if len(last_3_games) > 0:
            avg_yards = float(last_3_games['yards'].mean())
            avg_recs = float(last_3_games['receptions'].mean())
        else:
            avg_yards = None
            avg_recs = None

        lookup_rows.append({
            'player_id': player_id,
            'opponent': opponent,
            'season': season,
            'week': week,
            'wr_vs_team_avg_yards_last3': avg_yards,
            'wr_vs_team_receptions_last3': avg_recs
        })

    # Step 4: Create DataFrame and save
    print("\nStep 4: Creating lookup table...")
    lookup_df = pl.DataFrame(lookup_rows)

    print(f"Lookup table size: {len(lookup_df):,} rows")
    print(f"Columns: {lookup_df.columns}")

    # Save to parquet
    output_file = Path(CACHE_DIR) / "ml_training_data" / "matchup_lookup.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lookup_df.write_parquet(output_file)

    file_size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"\nSaved to: {output_file}")
    print(f"File size: {file_size_mb:.1f} MB")

    # Summary stats
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    non_null_yards = lookup_df['wr_vs_team_avg_yards_last3'].drop_nulls()
    non_null_recs = lookup_df['wr_vs_team_receptions_last3'].drop_nulls()

    print(f"Total matchups: {len(lookup_df):,}")
    print(f"With historical data: {len(non_null_yards):,} ({len(non_null_yards)/len(lookup_df)*100:.1f}%)")
    print(f"Without historical data: {len(lookup_df) - len(non_null_yards):,} (first-time matchups)")

    if len(non_null_yards) > 0:
        print(f"\nHistorical matchup stats:")
        print(f"  Avg yards/game: {non_null_yards.mean():.1f} (range: {non_null_yards.min():.1f} - {non_null_yards.max():.1f})")
        print(f"  Avg recs/game: {non_null_recs.mean():.1f} (range: {non_null_recs.min():.1f} - {non_null_recs.max():.1f})")

    print("\n" + "="*80)
    print("LOOKUP TABLE BUILD COMPLETE")
    print("="*80)
    print("\nNext step: Modify feature extraction to use this lookup table")


if __name__ == "__main__":
    main()
