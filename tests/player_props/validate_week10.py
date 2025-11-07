"""
Validate Week 10 2024 projections against actual outcomes
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.prop_projection_engine import generate_projection
from modules.prop_types import get_prop_types_for_position
import polars as pl

def validate_player_projections(player_id: str, player_name: str, position: str, season: int, week: int):
    """
    Generate projections for a player and compare to actual Week N performance.

    Args:
        player_id: Player GSIS ID
        player_name: Player display name
        position: Position
        season: Season year
        week: Week to validate
    """
    print(f"\n{'='*60}")
    print(f"{player_name} - Week {week} Validation")
    print('='*60)

    # Load actual stats for the week
    stats_file = Path(f"cache/positional_player_stats/{position.lower()}/{position.lower()}-{season}.csv")
    if not stats_file.exists():
        print(f"[SKIP] Stats file not found: {stats_file}")
        return

    df = pl.read_csv(stats_file)
    actual = df.filter(
        (pl.col('player_id') == player_id) &
        (pl.col('week') == week)
    )

    if len(actual) == 0:
        print(f"[SKIP] No Week {week} data found for {player_name}")
        return

    # Generate projections (using data through week-1)
    prop_types = get_prop_types_for_position(position)

    print(f"\nProjected vs Actual:")
    print(f"{'Prop Type':<20} {'Projected':<12} {'Actual':<12} {'Diff':<12} {'Accuracy'}")
    print('-'*70)

    for prop_type in prop_types:
        projection = generate_projection(
            player_id=player_id,
            season=season,
            week=week,
            position=position,
            prop_type=prop_type
        )

        if not projection or projection['final_projection'] == 0:
            continue

        # Get stat column for this prop type
        from modules.prop_types import get_stat_column_for_prop
        stat_col = get_stat_column_for_prop(prop_type)

        if stat_col not in actual.columns:
            continue

        projected = projection['final_projection']
        actual_value = actual[stat_col][0]
        diff = projected - actual_value

        # Calculate accuracy (percentage of actual)
        if actual_value > 0:
            accuracy_pct = (projected / actual_value) * 100
            accuracy_str = f"{accuracy_pct:.1f}%"
        else:
            accuracy_str = "N/A"

        print(f"{prop_type:<20} {projected:<12.1f} {actual_value:<12.1f} {diff:<12.1f} {accuracy_str}")


def main():
    """Run validation on multiple players."""
    print("="*60)
    print("WEEK 10 2024 PROJECTION VALIDATION")
    print("="*60)
    print("\nComparing projected values (using data through Week 9)")
    print("against actual Week 10 performance")

    # Load QB stats to find players
    qb_stats = pl.read_csv("cache/positional_player_stats/qb/qb-2024.csv")

    # Get QBs who played in Week 10
    week10_qbs = qb_stats.filter(pl.col('week') == 10).sort('passing_yards', descending=True).head(5)

    print(f"\nTesting top 5 QBs by passing yards in Week 10:")
    for i in range(len(week10_qbs)):
        player_name = week10_qbs['player_display_name'][i]
        player_id = week10_qbs['player_id'][i]
        print(f"  - {player_name} ({player_id})")

    # Validate each QB
    for i in range(len(week10_qbs)):
        player_name = week10_qbs['player_display_name'][i]
        player_id = week10_qbs['player_id'][i]

        validate_player_projections(
            player_id=player_id,
            player_name=player_name,
            position='QB',
            season=2024,
            week=10
        )

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nNote: Projections use weighted rolling average through Week 9")
    print("with sample size dampening. Adjustments are currently placeholders (1.0x).")
    print("Accuracy should improve once adjustment functions are implemented.")


if __name__ == "__main__":
    main()
