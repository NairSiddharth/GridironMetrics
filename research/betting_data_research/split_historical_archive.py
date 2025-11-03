"""
Split Historical Betting Archive

Splits the monolithic nfl_archive_10Y.json (2011-2021) into individual season files
matching the format used by the ESPN scraper.
"""

import json
from pathlib import Path
from collections import defaultdict

def split_archive():
    """Split historical archive into individual season files"""

    print("=== Splitting Historical Archive ===\n")

    # Load the monolithic archive
    archive_file = Path("cache/nfl_archive_10Y.json")
    if not archive_file.exists():
        print(f"ERROR: {archive_file} not found")
        return

    print(f"Loading {archive_file}...")
    with open(archive_file, 'r') as f:
        all_games = json.load(f)

    print(f"  Loaded {len(all_games)} total games\n")

    # Group by season
    games_by_season = defaultdict(list)
    for game in all_games:
        season = game.get('season')
        if season:
            games_by_season[season].append(game)

    # Create output directory
    output_dir = Path("cache/betting_lines")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each season
    print("Splitting into season files...")
    for season in sorted(games_by_season.keys()):
        games = games_by_season[season]
        output_file = output_dir / f"{season}.json"

        with open(output_file, 'w') as f:
            json.dump(games, f, indent=2)

        print(f"  {season}: {len(games)} games -> {output_file}")

    print(f"\nComplete! Created {len(games_by_season)} season files")

    # Create metadata
    metadata = {
        "source": "Historical archive (sportsbookreview) + ESPN API",
        "coverage": {
            "2011-2021": "sportsbookreview archive (split from nfl_archive_10Y.json)",
            "2022+": "ESPN public API"
        },
        "last_updated": "2024-11-02",
        "seasons": list(sorted(games_by_season.keys())),
        "total_games": sum(len(g) for g in games_by_season.values())
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created {metadata_file}")


if __name__ == "__main__":
    split_archive()
