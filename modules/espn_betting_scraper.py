"""
ESPN Betting Data Scraper

Scrapes NFL betting lines from ESPN's public API for historical and current seasons.
Outputs one JSON file per season in standardized format matching historical archive.
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ESPNBettingScraper:
    """Scraper for NFL betting data from ESPN's public API"""

    # Team name mapping: ESPN full names → standardized short names
    TEAM_NAME_MAP = {
        # AFC East
        "Buffalo Bills": "Bills",
        "Miami Dolphins": "Dolphins",
        "New England Patriots": "Patriots",
        "New York Jets": "Jets",

        # AFC North
        "Baltimore Ravens": "Ravens",
        "Cincinnati Bengals": "Bengals",
        "Cleveland Browns": "Browns",
        "Pittsburgh Steelers": "Steelers",

        # AFC South
        "Houston Texans": "Texans",
        "Indianapolis Colts": "Colts",
        "Jacksonville Jaguars": "Jaguars",
        "Tennessee Titans": "Titans",

        # AFC West
        "Denver Broncos": "Broncos",
        "Kansas City Chiefs": "Chiefs",
        "Las Vegas Raiders": "Raiders",
        "Los Angeles Chargers": "Chargers",

        # NFC East
        "Dallas Cowboys": "Cowboys",
        "New York Giants": "Giants",
        "Philadelphia Eagles": "Eagles",
        "Washington Commanders": "Commanders",
        "Washington Football Team": "Commanders",  # Historical name
        "Washington Redskins": "Commanders",  # Historical name

        # NFC North
        "Chicago Bears": "Bears",
        "Detroit Lions": "Lions",
        "Green Bay Packers": "Packers",
        "Minnesota Vikings": "Vikings",

        # NFC South
        "Atlanta Falcons": "Falcons",
        "Carolina Panthers": "Panthers",
        "New Orleans Saints": "Saints",
        "Tampa Bay Buccaneers": "Buccaneers",

        # NFC West
        "Arizona Cardinals": "Cardinals",
        "Los Angeles Rams": "Rams",
        "San Francisco 49ers": "49ers",
        "Seattle Seahawks": "Seahawks",

        # Historical team names
        "St. Louis Rams": "Rams",
        "San Diego Chargers": "Chargers",
        "Oakland Raiders": "Raiders",
    }

    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize scraper

        Args:
            rate_limit_delay: Seconds to wait between API requests (be respectful)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (GridironMetrics NFL Analysis)'
        })

    def _normalize_team_name(self, espn_name: str) -> str:
        """Convert ESPN team name to standardized short name"""
        return self.TEAM_NAME_MAP.get(espn_name, espn_name)

    def _convert_date_to_int(self, date_str: str) -> int:
        """Convert ISO date string to YYYYMMDD integer"""
        # ESPN dates are like: "2022-09-08T20:20Z"
        date_part = date_str.split('T')[0]  # Get "2022-09-08"
        return int(date_part.replace('-', ''))  # Convert to 20220908

    def get_games_for_season(self, season: int) -> List[Dict]:
        """
        Get all games for a given season

        Args:
            season: Year (e.g., 2022)

        Returns:
            List of game dictionaries with IDs and basic info
        """
        logger.info(f"Fetching games for {season} season...")

        # NFL regular season typically runs September through January
        # We'll query week by week to be thorough
        all_games = []

        # Season dates: Sept through Feb (covers regular + playoffs)
        start_date = f"{season}0901"
        end_date = f"{season + 1}0228"

        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {
            'dates': f"{start_date}-{end_date}",
            'limit': 1000  # Get all games in range
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            games = data.get('events', [])
            logger.info(f"  Found {len(games)} games")

            for game in games:
                all_games.append({
                    'id': game['id'],
                    'name': game['name'],
                    'date': game['date'],
                    'season': season
                })

            time.sleep(self.rate_limit_delay)

        except Exception as e:
            logger.error(f"Error fetching games for {season}: {e}")

        return all_games

    def get_odds_for_game(self, game_id: str) -> Optional[Dict]:
        """
        Get betting odds for a specific game

        Args:
            game_id: ESPN game ID

        Returns:
            Dict with odds data or None if unavailable
        """
        url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{game_id}/competitions/{game_id}/odds"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            time.sleep(self.rate_limit_delay)

            # ESPN returns multiple bookmakers - we'll use the first one with complete data
            items = data.get('items', [])
            if not items:
                return None

            # Try to find ESPN BET or consensus odds (provider ID 58 or 45)
            preferred_providers = ['58', '45']  # ESPN BET, Consensus
            best_odds = None

            for item in items:
                provider_id = item.get('provider', {}).get('id', '')
                if provider_id in preferred_providers or best_odds is None:
                    best_odds = item
                    if provider_id in preferred_providers:
                        break

            if not best_odds:
                return None

            return best_odds

        except Exception as e:
            logger.debug(f"No odds available for game {game_id}: {e}")
            return None

    def get_game_details(self, game_id: str) -> Optional[Dict]:
        """
        Get detailed game information including final scores

        Args:
            game_id: ESPN game ID

        Returns:
            Dict with game details or None
        """
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
        params = {'event': game_id}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            time.sleep(self.rate_limit_delay)

            # Extract team info and scores
            header = data.get('header', {})
            competitions = header.get('competitions', [{}])[0]
            competitors = competitions.get('competitors', [])

            if len(competitors) != 2:
                return None

            # Determine home/away
            home_team = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away_team = next((c for c in competitors if c.get('homeAway') == 'away'), None)

            if not home_team or not away_team:
                return None

            return {
                'home_team': home_team.get('team', {}).get('displayName', ''),
                'away_team': away_team.get('team', {}).get('displayName', ''),
                'home_score': home_team.get('score', '0'),
                'away_score': away_team.get('score', '0')
            }

        except Exception as e:
            logger.debug(f"Could not get game details for {game_id}: {e}")
            return None

    def scrape_season(self, season: int) -> List[Dict]:
        """
        Scrape complete betting data for a season

        Args:
            season: Year (e.g., 2022)

        Returns:
            List of game records in standardized format
        """
        logger.info(f"\n=== Scraping {season} Season ===")

        games = self.get_games_for_season(season)
        if not games:
            logger.warning(f"No games found for {season}")
            return []

        records = []
        games_with_odds = 0

        for i, game in enumerate(games, 1):
            logger.info(f"Processing game {i}/{len(games)}: {game['name']}")

            # Get odds
            odds = self.get_odds_for_game(game['id'])
            if not odds:
                logger.warning(f"  No odds available - skipping")
                continue

            # Get game details (teams, scores)
            details = self.get_game_details(game['id'])
            if not details:
                logger.warning(f"  Could not get game details - skipping")
                continue

            # Parse odds data
            spread = odds.get('spread', 0)
            over_under = odds.get('overUnder', 0)

            away_odds = odds.get('awayTeamOdds', {})
            home_odds = odds.get('homeTeamOdds', {})

            # Determine which team is favored
            away_favorite = away_odds.get('favorite', False)

            # Get closing lines (preferred) or current lines
            away_close = away_odds.get('close', {})
            home_close = home_odds.get('close', {})

            away_ml = away_close.get('moneyLine', {}).get('american', away_odds.get('moneyLine', 0))
            home_ml = home_close.get('moneyLine', {}).get('american', home_odds.get('moneyLine', 0))

            # Handle 'EVEN' moneylines (convert to +100)
            if away_ml == 'EVEN':
                away_ml = 100
            if home_ml == 'EVEN':
                home_ml = 100

            # Spread is always relative to away team in ESPN data
            if away_favorite:
                home_spread = abs(spread)
                away_spread = -abs(spread)
            else:
                away_spread = abs(spread)
                home_spread = -abs(spread)

            # Create standardized record
            record = {
                'season': season,
                'date': self._convert_date_to_int(game['date']),
                'home_team': self._normalize_team_name(details['home_team']),
                'away_team': self._normalize_team_name(details['away_team']),
                'home_1stQtr': '0',  # ESPN doesn't provide quarter scores in this endpoint
                'away_1stQtr': '0',
                'home_2ndQtr': '0',
                'away_2ndQtr': '0',
                'home_3rdQtr': '0',
                'away_3rdQtr': '0',
                'home_4thQtr': '0',
                'away_4thQtr': '0',
                'home_final': str(details['home_score']),
                'away_final': str(details['away_score']),
                'home_close_ml': int(home_ml) if home_ml else 0,
                'away_close_ml': int(away_ml) if away_ml else 0,
                'home_open_spread': float(home_spread),
                'away_open_spread': float(away_spread),
                'home_close_spread': float(home_spread),  # ESPN doesn't differentiate open/close clearly
                'away_close_spread': float(away_spread),
                'home_2H_spread': 0.0,  # Not available in ESPN data
                'away_2H_spread': 0.0,
                '2H_total': 0.0,
                'open_over_under': float(over_under) if over_under else 0.0,
                'close_over_under': float(over_under) if over_under else 0.0,
            }

            records.append(record)
            games_with_odds += 1
            logger.info(f"  ✓ Scraped: {record['away_team']} @ {record['home_team']} (Spread: {away_spread})")

        logger.info(f"\n{season} Summary: {games_with_odds}/{len(games)} games with complete odds data")
        return records

    def save_season_data(self, season: int, records: List[Dict], output_dir: Path):
        """
        Save season data to JSON file

        Args:
            season: Year
            records: List of game records
            output_dir: Directory to save to
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{season}.json"

        with open(output_file, 'w') as f:
            json.dump(records, f, indent=2)

        logger.info(f"Saved {len(records)} games to {output_file}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Example usage
    scraper = ESPNBettingScraper(rate_limit_delay=0.5)

    # Test on 2022
    output_dir = Path("cache/betting_lines")
    records = scraper.scrape_season(2022)
    if records:
        scraper.save_season_data(2022, records, output_dir)
