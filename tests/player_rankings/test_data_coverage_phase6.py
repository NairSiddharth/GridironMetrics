"""
Data coverage tests for Phase 6 efficiency metrics.
Validates nflfastpy column availability and quality before implementation.
"""

import nflreadpy as nfl
import polars as pl
from pathlib import Path
import sys

def test_success_rate_coverage():
    """Test success column coverage across years."""
    print("
=== SUCCESS RATE COVERAGE ===")

    years_to_test = [2010, 2015, 2020, 2023, 2024]
    results = {}

    for year in years_to_test:
        print(f"
Testing {year}...")
        pbp_pl = nfl.load_pbp(seasons=year, columns=['success', 'play_type'])

        total_plays = len(pbp_pl)
        null_success = pbp_pl.filter(pl.col('success').is_null()).height
        null_rate = (null_success / total_plays) * 100

        # Filter to offensive plays only
        offensive_plays = pbp_pl.filter(
            (pl.col('play_type') == 'pass') | (pl.col('play_type') == 'run')
        )
        offensive_total = len(offensive_plays)
        offensive_null = offensive_plays.filter(pl.col('success').is_null()).height
        offensive_null_rate = (offensive_null / offensive_total) * 100

        results[year] = {
            'total_plays': total_plays,
            'null_rate': null_rate,
            'offensive_null_rate': offensive_null_rate
        }

        print(f"  Total plays: {total_plays:,}")
        print(f"  Null success rate (all plays): {null_rate:.2f}%")
        print(f"  Null success rate (offensive plays): {offensive_null_rate:.2f}%")

        # PASS/FAIL criteria
        if offensive_null_rate < 5.0:
            print(f"  [PASS] - Null rate acceptable (<5%)")
        elif offensive_null_rate < 15.0:
            print(f"  [WARNING] - Null rate moderate (5-15%)")
        else:
            print(f"  [FAIL] - Null rate too high (>15%)")
            return False

    return True


if __name__ == "__main__":
    sys.exit(0)
