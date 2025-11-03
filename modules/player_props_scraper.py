"""
NFL Player Props Scraper using The Odds API

Fetches player prop betting lines from DFS sites (PrizePicks, Underdog Fantasy)
for current/upcoming NFL games. Organizes data by player/week/day for easy
comparison with player rankings.

Also supports historical data fetching for backtesting projections against
actual betting lines from May 3, 2023 onwards (when player props data became available).
"""
import requests
import json
import os
import time
import csv
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class PlayerPropsScraper:
    """Scraper for NFL player prop betting lines"""

    # Available prop markets
    MARKETS = {
        'player_pass_yds': 'Passing Yards',
        'player_pass_tds': 'Passing Touchdowns',
        'player_rush_yds': 'Rushing Yards',
        'player_receptions': 'Receptions',
        'player_reception_yds': 'Receiving Yards',
    }

    # DFS bookmakers that offer props
    BOOKMAKERS = ['prizepicks', 'underdog']

    # NFL season start dates (Week 1 Sunday) for historical data
    # Used to calculate Tuesday/Friday timestamps for each week
    # NOTE: Historical player props available from May 3, 2023 onwards
    #       2023 data available from ~Week 10+, 2024+ full seasons available
    SEASON_START_DATES = {
        2020: datetime(2020, 9, 13, tzinfo=timezone.utc),  # Week 1 Sunday (no props data)
        2021: datetime(2021, 9, 12, tzinfo=timezone.utc),  # (no props data)
        2022: datetime(2022, 9, 11, tzinfo=timezone.utc),  # (no props data)
        2023: datetime(2023, 9, 10, tzinfo=timezone.utc),  # Props from Week 10+ only
        2024: datetime(2024, 9, 8, tzinfo=timezone.utc),   # Full season available
        2025: datetime(2025, 9, 7, tzinfo=timezone.utc),   # Full season available
    }

    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 1.0):
        """
        Initialize scraper

        Args:
            api_key: The Odds API key (reads from .env if not provided)
            rate_limit_delay: Seconds to wait between API calls
        """
        if api_key:
            self.api_key = api_key
        else:
            load_dotenv()
            # Try PAID_ODDS_API first (for historical data), then fall back to FREE_ODDS_API or ODDS_API
            self.api_key = os.getenv('PAID_ODDS_API') or os.getenv('ODDS_API') or os.getenv('FREE_ODDS_API')
            if not self.api_key:
                raise ValueError("ODDS_API key not found in environment or .env file (tried PAID_ODDS_API, ODDS_API, FREE_ODDS_API)")

        self.rate_limit_delay = rate_limit_delay
        self.base_url = 'https://api.the-odds-api.com/v4'
        self.sport = 'americanfootball_nfl'
        self.requests_used = 0
        self.requests_remaining = None

        # Player lookup cache (loaded on demand)
        self._player_lookup = None
        self._qualifying_players = None

    def _load_rosters(self, season: int) -> Dict[str, Dict]:
        """
        Load roster data to build player lookup

        Args:
            season: Year (e.g., 2025)

        Returns:
            Dict mapping full_name to {gsis_id, position, team, display_name}
        """
        roster_file = Path(f'cache/rosters/rosters-{season}.csv')
        if not roster_file.exists():
            logger.warning(f"Roster file not found: {roster_file}")
            return {}

        player_lookup = {}
        with open(roster_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_name = row.get('full_name', '').strip()
                first_name = row.get('first_name', '').strip()
                last_name = row.get('last_name', '').strip()

                if not full_name or not first_name or not last_name:
                    continue

                # Create display name: "D.Prescott"
                display_name = f"{first_name[0]}.{last_name}"

                # Store player info (use latest entry per player)
                player_lookup[full_name.lower()] = {
                    'gsis_id': row.get('gsis_id', ''),
                    'position': row.get('position', ''),
                    'team': row.get('team', ''),
                    'display_name': display_name,
                    'full_name': full_name,
                    'first_name': first_name,
                    'last_name': last_name
                }

        logger.info(f"Loaded {len(player_lookup)} players from {season} roster")
        return player_lookup

    def _normalize_player_name(self, full_name: str) -> str:
        """
        Normalize player name for folder structure

        Args:
            full_name: Full name like "Dak Prescott"

        Returns:
            Normalized name like "D_Prescott"
        """
        parts = full_name.strip().split()
        if len(parts) < 2:
            return full_name.replace(' ', '_')

        first_initial = parts[0][0].upper()
        last_name = parts[-1]

        # Remove special characters from last name
        last_name = re.sub(r'[^\w\s-]', '', last_name)

        return f"{first_initial}_{last_name}"

    def _get_qualifying_players(self, season: int) -> Set[str]:
        """
        Get set of qualifying player names from ranking outputs

        Args:
            season: Year (e.g., 2025)

        Returns:
            Set of display names like {"P.Mahomes", "D.Prescott", ...}
        """
        qualifying = set()
        output_dir = Path(f'output/{season}')

        if not output_dir.exists():
            logger.warning(f"Output directory not found: {output_dir}")
            return qualifying

        # Check all position ranking files
        for position in ['qb', 'rb', 'wr', 'te']:
            ranking_file = output_dir / f'{position}_rankings.md'
            if not ranking_file.exists():
                continue

            with open(ranking_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # Parse markdown table lines: "| 1    | P.Mahomes    | KC   |"
                    if '|' not in line or line.startswith('|:') or line.startswith('| Rank'):
                        continue

                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) < 4:
                        continue

                    # Player name is in column 2 (index 2 after split)
                    player_name = parts[2].strip()
                    if player_name and '.' in player_name:  # Format check: "P.Mahomes"
                        qualifying.add(player_name)

        logger.info(f"Found {len(qualifying)} qualifying players from {season} rankings")
        return qualifying

    def _match_player_to_roster(self, api_name: str, roster_lookup: Dict[str, Dict]) -> Optional[Dict]:
        """
        Match API player name to roster entry

        Args:
            api_name: Name from API like "Dak Prescott"
            roster_lookup: Roster lookup dictionary

        Returns:
            Player info dict or None if no match
        """
        # Try exact match first (case-insensitive)
        api_name_lower = api_name.lower().strip()
        if api_name_lower in roster_lookup:
            return roster_lookup[api_name_lower]

        # Try variations (e.g., "D.J. Moore" vs "DJ Moore")
        # Remove periods and extra spaces
        normalized = api_name_lower.replace('.', '').replace('  ', ' ')

        for roster_name, player_info in roster_lookup.items():
            roster_normalized = roster_name.replace('.', '').replace('  ', ' ')
            if normalized == roster_normalized:
                return player_info

        # Log unmatched
        logger.debug(f"Could not match player: {api_name}")
        return None

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make API request with rate limiting and error handling

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data
        """
        params['api_key'] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        logger.debug(f"Making request to {url}")
        time.sleep(self.rate_limit_delay)

        response = requests.get(url, params=params, timeout=30)

        # Track API usage
        self.requests_remaining = response.headers.get('x-requests-remaining')
        self.requests_used = response.headers.get('x-requests-used')

        if response.status_code != 200:
            logger.error(f"API request failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            response.raise_for_status()

        return response.json()

    def get_nfl_events(self) -> List[Dict]:
        """
        Get list of upcoming NFL events

        Returns:
            List of event dictionaries with game info
        """
        logger.info("Fetching NFL events...")
        endpoint = f'sports/{self.sport}/events'
        events = self._make_request(endpoint, {})
        logger.info(f"Found {len(events)} upcoming NFL games")
        return events

    def get_player_props(
        self,
        event_id: str,
        markets: Optional[List[str]] = None,
        bookmakers: Optional[List[str]] = None
    ) -> Dict:
        """
        Get player props for a specific game

        Args:
            event_id: The Odds API event ID
            markets: List of markets to fetch (defaults to all)
            bookmakers: List of bookmakers (defaults to prizepicks, underdog)

        Returns:
            Dictionary with game info and player props
        """
        if markets is None:
            markets = list(self.MARKETS.keys())
        if bookmakers is None:
            bookmakers = self.BOOKMAKERS

        markets_str = ','.join(markets)
        bookmakers_str = ','.join(bookmakers)

        endpoint = f'sports/{self.sport}/events/{event_id}/odds'
        params = {
            'markets': markets_str,
            'bookmakers': bookmakers_str,
            'oddsFormat': 'american',
        }

        logger.debug(f"Fetching props for event {event_id}")
        return self._make_request(endpoint, params)

    def parse_props(self, event_data: Dict) -> List[Dict]:
        """
        Parse raw API response into structured prop records

        Args:
            event_data: Raw API response for a single event

        Returns:
            List of prop records
        """
        props = []

        game_info = {
            'game_id': event_data.get('id'),
            'home_team': event_data.get('home_team'),
            'away_team': event_data.get('away_team'),
            'commence_time': event_data.get('commence_time'),
        }

        for bookmaker in event_data.get('bookmakers', []):
            bookmaker_key = bookmaker.get('key')

            for market in bookmaker.get('markets', []):
                market_key = market.get('key')
                last_update = market.get('last_update')

                # Group outcomes by player
                player_props = {}
                for outcome in market.get('outcomes', []):
                    player = outcome.get('description')
                    name = outcome.get('name')  # 'Over' or 'Under'
                    price = outcome.get('price')
                    point = outcome.get('point')

                    if player not in player_props:
                        player_props[player] = {
                            'player': player,
                            'market': market_key,
                            'market_name': self.MARKETS.get(market_key, market_key),
                            'line': point,
                            'bookmaker': bookmaker_key,
                            'last_update': last_update,
                        }

                    # Add over/under prices
                    if name == 'Over':
                        player_props[player]['over_price'] = price
                    elif name == 'Under':
                        player_props[player]['under_price'] = price

                # Add game info to each prop
                for prop in player_props.values():
                    prop.update(game_info)
                    props.append(prop)

        return props

    def scrape_week_props(
        self,
        markets: Optional[List[str]] = None,
        filter_players: Optional[Set[str]] = None
    ) -> List[Dict]:
        """
        Scrape player props for all upcoming games this week

        Args:
            markets: List of markets to fetch (defaults to all)
            filter_players: Optional set of player names to filter for

        Returns:
            List of all player prop records
        """
        logger.info("Starting week props scrape...")

        # Get list of games
        events = self.get_nfl_events()

        if not events:
            logger.warning("No upcoming events found")
            return []

        all_props = []

        for i, event in enumerate(events, 1):
            game_desc = f"{event['away_team']} @ {event['home_team']}"
            logger.info(f"Processing game {i}/{len(events)}: {game_desc}")

            try:
                # Get props for this game
                event_data = self.get_player_props(
                    event['id'],
                    markets=markets
                )

                # Parse into structured records
                props = self.parse_props(event_data)

                # Filter by players if specified
                if filter_players:
                    props = [p for p in props if p['player'] in filter_players]
                    logger.info(f"  Found {len(props)} props for tracked players")
                else:
                    logger.info(f"  Found {len(props)} total props")

                all_props.extend(props)

            except Exception as e:
                logger.error(f"  Error processing game: {e}")
                continue

        logger.info(f"\nScrape complete! Total props: {len(all_props)}")
        logger.info(f"API requests remaining: {self.requests_remaining}")

        return all_props

    def save_props(
        self,
        props: List[Dict],
        output_dir: str = 'cache/player_props',
        week: Optional[int] = None
    ) -> Path:
        """
        Save props to JSON file

        Args:
            props: List of prop records
            output_dir: Directory to save output
            week: Optional week number for filename

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if week:
            filename = output_path / f'nfl_props_week{week}_{timestamp}.json'
        else:
            filename = output_path / f'nfl_props_{timestamp}.json'

        # Add metadata
        data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_props': len(props),
                'api_requests_used': self.requests_used,
                'api_requests_remaining': self.requests_remaining,
                'markets': list(self.MARKETS.keys()),
                'bookmakers': self.BOOKMAKERS,
            },
            'props': props
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(props)} props to {filename}")
        return filename

    def save_props_by_player(
        self,
        props: List[Dict],
        season: int,
        week: int,
        day: str,  # 'tuesday' or 'friday'
        filter_to_ranked: bool = True,
        output_dir: str = 'cache/player_props'
    ) -> Dict[str, Path]:
        """
        Save props organized by player/week/day structure

        Args:
            props: List of prop records from scraping
            season: Year (e.g., 2025)
            week: Week number
            day: 'tuesday' or 'friday'
            filter_to_ranked: Only save players in ranking outputs
            output_dir: Base directory for player props

        Returns:
            Dict mapping player_name -> saved file path
        """
        # Load roster and qualifying players
        if self._player_lookup is None:
            self._player_lookup = self._load_rosters(season)

        if filter_to_ranked and self._qualifying_players is None:
            self._qualifying_players = self._get_qualifying_players(season)

        # Group props by player
        by_player = defaultdict(list)
        matched_count = 0
        unmatched_count = 0

        for prop in props:
            player_api_name = prop.get('player', '')
            if not player_api_name:
                continue

            # Match to roster
            player_info = self._match_player_to_roster(player_api_name, self._player_lookup)
            if not player_info:
                unmatched_count += 1
                continue

            matched_count += 1

            # Check if player qualifies
            if filter_to_ranked:
                if player_info['display_name'] not in self._qualifying_players:
                    continue

            # Add player info to prop
            prop['gsis_id'] = player_info['gsis_id']
            prop['position'] = player_info['position']
            prop['team'] = player_info['team']
            prop['display_name'] = player_info['display_name']

            # Group by player
            by_player[player_api_name].append(prop)

        logger.info(f"Matched {matched_count}/{matched_count + unmatched_count} players to roster")
        logger.info(f"Saving props for {len(by_player)} players")

        # Save each player's props
        saved_files = {}
        output_path = Path(output_dir) / str(season)

        for player_api_name, player_props in by_player.items():
            # Get normalized folder name
            normalized_name = self._normalize_player_name(player_api_name)

            # Create player/week directory
            player_dir = output_path / normalized_name / f'week{week}'
            player_dir.mkdir(parents=True, exist_ok=True)

            # Save to day file
            filename = player_dir / f'{day}.json'

            # Organize props by market type
            organized_props = {
                'player': player_api_name,
                'gsis_id': player_props[0].get('gsis_id', ''),
                'position': player_props[0].get('position', ''),
                'team': player_props[0].get('team', ''),
                'display_name': player_props[0].get('display_name', ''),
                'game': {
                    'game_id': player_props[0].get('game_id', ''),
                    'home_team': player_props[0].get('home_team', ''),
                    'away_team': player_props[0].get('away_team', ''),
                    'commence_time': player_props[0].get('commence_time', '')
                },
                'props': {},
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'season': season,
                    'week': week,
                    'day': day,
                    'bookmaker': player_props[0].get('bookmaker', ''),
                    'last_update': player_props[0].get('last_update', '')
                }
            }

            # Group props by market
            for prop in player_props:
                market_key = prop.get('market', '')
                market_name = prop.get('market_name', market_key)

                organized_props['props'][market_name] = {
                    'line': prop.get('line'),
                    'over_price': prop.get('over_price'),
                    'under_price': prop.get('under_price')
                }

            # Write file
            with open(filename, 'w') as f:
                json.dump(organized_props, f, indent=2)

            saved_files[player_api_name] = filename
            logger.debug(f"  Saved {player_api_name} -> {filename}")

        logger.info(f"Saved {len(saved_files)} player files to {output_path}")
        return saved_files

    def scrape_and_save_by_player(
        self,
        season: int,
        week: int,
        day: str,  # 'tuesday' or 'friday'
        filter_to_ranked: bool = True,
        markets: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Scrape current week props and save by player/week/day

        Args:
            season: Year (e.g., 2025)
            week: Week number
            day: 'tuesday' or 'friday'
            filter_to_ranked: Only save players in ranking outputs
            markets: List of markets to fetch (defaults to all)

        Returns:
            Dict mapping player_name -> saved file path
        """
        logger.info(f"\n=== Scraping {season} Week {week} ({day.title()}) ===")

        # Scrape all props
        props = self.scrape_week_props(markets=markets)

        if not props:
            logger.warning("No props found")
            return {}

        # Save by player
        saved_files = self.save_props_by_player(
            props=props,
            season=season,
            week=week,
            day=day,
            filter_to_ranked=filter_to_ranked
        )

        logger.info(f"\n{day.title()} scrape complete!")
        logger.info(f"API requests remaining: {self.requests_remaining}")

        return saved_files

    @staticmethod
    def calculate_nfl_week_timestamp(season: int, week: int, day: str) -> str:
        """
        Calculate timestamp for Tuesday or Friday of a given NFL week.

        Args:
            season: Year (e.g., 2024)
            week: Week number (1-18)
            day: 'tuesday' or 'friday'

        Returns:
            ISO 8601 timestamp string (e.g., '2024-09-10T13:00:00Z')

        Notes:
            - Tuesday 9am ET = 13:00 UTC (opening lines)
            - Friday 9am ET = 13:00 UTC (post-injury report lines)
            - Calculates from Week 1 Sunday + (week-1)*7 days
        """
        if season not in PlayerPropsScraper.SEASON_START_DATES:
            raise ValueError(f"Season {season} not in SEASON_START_DATES")

        if day.lower() not in ['tuesday', 'friday']:
            raise ValueError(f"Day must be 'tuesday' or 'friday', got '{day}'")

        # Get Week 1 Sunday for this season
        week1_sunday = PlayerPropsScraper.SEASON_START_DATES[season]

        # Calculate the Sunday of the target week
        # Week 1 = week1_sunday, Week 2 = week1_sunday + 7 days, etc.
        target_sunday = week1_sunday + timedelta(days=(week - 1) * 7)

        # Calculate Tuesday or Friday relative to Sunday
        if day.lower() == 'tuesday':
            # Tuesday = Sunday + 2 days
            target_date = target_sunday + timedelta(days=2)
        else:  # friday
            # Friday = Sunday + 5 days
            target_date = target_sunday + timedelta(days=5)

        # Set time to 9am ET = 13:00 UTC (or 14:00 UTC during DST)
        # For simplicity, always use 13:00 UTC
        target_datetime = target_date.replace(hour=13, minute=0, second=0, microsecond=0)

        # Return ISO 8601 format required by The Odds API
        return target_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')

    def get_historical_events(self, date: str) -> List[Dict]:
        """
        Get historical events (game list) for a specific timestamp.

        Args:
            date: ISO 8601 timestamp (e.g., '2024-09-10T13:00:00Z')

        Returns:
            List of event dictionaries with game info (id, teams, commence_time)

        Notes:
            - Uses: /v4/historical/sports/americanfootball_nfl/events
            - Cost: 1 credit per request
            - Returns events that existed at the specified timestamp
            - Required first step before fetching historical event odds
        """
        endpoint = f'historical/sports/{self.sport}/events'
        params = {'date': date}

        logger.debug(f"Fetching historical events for {date}")
        data = self._make_request(endpoint, params)

        # Historical endpoint returns {'data': [...events...]}
        if isinstance(data, dict) and 'data' in data:
            events = data['data']
        else:
            events = data if isinstance(data, list) else []

        logger.info(f"Found {len(events)} NFL events at {date}")
        return events

    def get_historical_event_odds(
        self,
        event_id: str,
        date: str,
        markets: Optional[List[str]] = None,
        bookmakers: Optional[List[str]] = None
    ) -> Dict:
        """
        Get historical odds for a single event at a specific timestamp.

        Args:
            event_id: The Odds API event ID
            date: ISO 8601 timestamp (e.g., '2024-09-10T13:00:00Z')
            markets: List of markets to fetch (defaults to all player props)
            bookmakers: List of bookmakers (defaults to prizepicks, underdog)

        Returns:
            Event dictionary with odds data

        Notes:
            - Uses: /v4/historical/sports/americanfootball_nfl/events/{id}/odds
            - Cost: 10 credits per request (per event)
            - Player props available from May 3, 2023 onwards
            - Returns snapshot nearest to (but not after) specified timestamp
        """
        if markets is None:
            markets = list(self.MARKETS.keys())
        if bookmakers is None:
            bookmakers = self.BOOKMAKERS

        markets_str = ','.join(markets)
        bookmakers_str = ','.join(bookmakers)

        endpoint = f'historical/sports/{self.sport}/events/{event_id}/odds'
        params = {
            'date': date,
            'markets': markets_str,
            'bookmakers': bookmakers_str,
            'oddsFormat': 'american',
        }

        logger.debug(f"Fetching historical odds for event {event_id}")
        data = self._make_request(endpoint, params)

        # Historical endpoint returns {'data': {event_data}}
        if isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            return data

    def get_historical_odds(
        self,
        date: str,
        markets: Optional[List[str]] = None,
        bookmakers: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get historical odds for all NFL games at a specific timestamp.

        Two-step process required for player props:
        1. Get list of events at that timestamp
        2. Query each event individually for odds

        Args:
            date: ISO 8601 timestamp (e.g., '2024-09-10T13:00:00Z')
            markets: List of markets to fetch (defaults to all player props)
            bookmakers: List of bookmakers (defaults to prizepicks, underdog)

        Returns:
            List of event dictionaries with historical odds data

        Notes:
            - Player props available from May 3, 2023 onwards
            - Cost: 1 credit for events + 10 credits per event for odds
            - Example: 16 games = 1 + (16 × 10) = 161 credits total
            - Rate limited between event queries
        """
        # Step 1: Get event IDs (1 credit)
        events = self.get_historical_events(date)

        if not events:
            logger.warning(f"No events found at {date}")
            return []

        logger.info(f"Fetching odds for {len(events)} events (10 credits each)")

        # Step 2: Query each event for odds (10 credits × num_events)
        all_event_data = []

        for i, event in enumerate(events, 1):
            try:
                event_odds = self.get_historical_event_odds(
                    event_id=event['id'],
                    date=date,
                    markets=markets,
                    bookmakers=bookmakers
                )

                all_event_data.append(event_odds)

                # Progress logging
                if i % 5 == 0:
                    logger.info(f"  Progress: {i}/{len(events)} events fetched")

                # Rate limiting between events
                if i < len(events):
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"  Failed to fetch event {event['id']}: {e}")
                continue

        logger.info(
            f"Retrieved odds for {len(all_event_data)}/{len(events)} events "
            f"(~{1 + len(all_event_data) * 10} credits used)"
        )

        return all_event_data

    def fetch_historical_week_props(
        self,
        season: int,
        week: int,
        day: str,
        filter_to_ranked: bool = True
    ) -> Dict[str, Path]:
        """
        Fetch and save historical props for a specific week/day.

        Args:
            season: Year (e.g., 2024)
            week: Week number (1-18)
            day: 'tuesday' or 'friday'
            filter_to_ranked: Only save players in ranking outputs

        Returns:
            Dict mapping player_name -> saved file path

        Notes:
            - Calculates timestamp for Tuesday/Friday of target week
            - Fetches all props from that timestamp
            - Saves in player/week/day structure under cache/player_props/historical/
            - Uses DraftKings/Caesars for 2023 (only ones with historical data)
            - Uses PrizePicks/Underdog for 2024+ (DFS sites, better coverage)
        """
        logger.info(f"\n=== Fetching Historical: {season} Week {week} ({day.title()}) ===")

        # Select bookmakers based on season
        # 2023: Only DraftKings and Fliff (DFS sites) have historical data
        # 2024+: Use PrizePicks and Underdog Fantasy (DFS sites)
        if season <= 2023:
            bookmakers = ['draftkings', 'fliff']  # DFS sites with 2023 historical data
            logger.info(f"Using DFS sites for {season} data: {bookmakers}")
        else:
            bookmakers = self.BOOKMAKERS  # PrizePicks + Underdog
            logger.info(f"Using DFS sites for {season} data: {bookmakers}")

        # Calculate timestamp
        timestamp = self.calculate_nfl_week_timestamp(season, week, day)
        logger.info(f"Target timestamp: {timestamp}")

        # Fetch historical data with season-appropriate bookmakers
        events = self.get_historical_odds(timestamp, bookmakers=bookmakers)

        if not events:
            logger.warning(f"No historical data found for {timestamp}")
            return {}

        # Parse all props from all events
        all_props = []
        for event in events:
            props = self.parse_props(event)
            all_props.extend(props)

        logger.info(f"Parsed {len(all_props)} total props from {len(events)} games")

        # Save by player (same directory structure as current props)
        saved_files = self.save_props_by_player(
            props=all_props,
            season=season,
            week=week,
            day=day,
            filter_to_ranked=filter_to_ranked,
            output_dir='cache/player_props'
        )

        logger.info(f"Historical {day.title()} fetch complete!")
        logger.info(f"API requests remaining: {self.requests_remaining}")

        return saved_files

    def fetch_historical_season(
        self,
        season: int,
        start_week: int = 1,
        end_week: int = 18,
        days: List[str] = ['tuesday', 'friday'],
        filter_to_ranked: bool = True
    ) -> Dict[int, Dict[str, Dict[str, Path]]]:
        """
        Fetch historical props for an entire season.

        Args:
            season: Year (e.g., 2024)
            start_week: First week to fetch (default: 1)
            end_week: Last week to fetch (default: 18)
            days: Days to fetch (default: ['tuesday', 'friday'])
            filter_to_ranked: Only save players in ranking outputs

        Returns:
            Nested dict: {week: {day: {player: filepath}}}

        Notes:
            - Fetches Tuesday + Friday for each week (2 requests per week)
            - 18 weeks × 2 days = 36 API requests per season
            - Rate limited to avoid hitting API limits
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"FETCHING HISTORICAL SEASON: {season}")
        logger.info(f"{'='*70}")
        logger.info(f"Weeks {start_week}-{end_week}, Days: {', '.join(days)}")

        results = {}

        for week in range(start_week, end_week + 1):
            results[week] = {}

            for day in days:
                try:
                    saved_files = self.fetch_historical_week_props(
                        season=season,
                        week=week,
                        day=day,
                        filter_to_ranked=filter_to_ranked
                    )
                    results[week][day] = saved_files

                    logger.info(f"✓ Week {week} {day.title()}: {len(saved_files)} players saved")

                except Exception as e:
                    logger.error(f"✗ Week {week} {day.title()} failed: {e}")
                    results[week][day] = {}
                    continue

            # Rate limiting between weeks
            if week < end_week:
                time.sleep(self.rate_limit_delay)

        total_fetched = sum(
            len(results[w][d])
            for w in results
            for d in results[w]
        )

        logger.info(f"\n{'='*70}")
        logger.info(f"SEASON {season} COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total player files saved: {total_fetched}")
        logger.info(f"API requests remaining: {self.requests_remaining}")

        return results


def main():
    """Main function for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Scrape NFL player props')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    parser.add_argument('--week', type=int, default=11, help='Week number')
    parser.add_argument('--day', type=str, default='tuesday', choices=['tuesday', 'friday'],
                        help='Day of week to scrape')
    parser.add_argument('--all-players', action='store_true',
                        help='Save all players (not just ranked)')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy flat file format instead of player-based')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    scraper = PlayerPropsScraper(rate_limit_delay=1.0)

    try:
        if args.legacy:
            # Legacy mode: save all props to single file
            props = scraper.scrape_week_props()

            if not props:
                print("\nNo props found. Check if there are upcoming games.")
                return

            # Display summary
            print(f"\n{'='*60}")
            print(f"SCRAPED {len(props)} NFL PLAYER PROPS")
            print(f"{'='*60}\n")

            # Group by market
            by_market = defaultdict(list)
            for prop in props:
                by_market[prop['market_name']].append(prop)

            for market, market_props in by_market.items():
                print(f"\n{market}: {len(market_props)} props")
                # Show first 5
                for prop in market_props[:5]:
                    print(f"  {prop['player']}: {prop['line']} "
                          f"(O: {prop.get('over_price')}, U: {prop.get('under_price')})")
                if len(market_props) > 5:
                    print(f"  ... and {len(market_props) - 5} more")

            # Save to file
            filename = scraper.save_props(props, week=args.week)
            print(f"\nData saved to: {filename}")

        else:
            # New mode: save by player/week/day
            saved_files = scraper.scrape_and_save_by_player(
                season=args.season,
                week=args.week,
                day=args.day,
                filter_to_ranked=not args.all_players
            )

            print(f"\n{'='*60}")
            print(f"SAVED {len(saved_files)} PLAYER PROP FILES")
            print(f"{'='*60}\n")

            # Show sample files
            for i, (player, filepath) in enumerate(list(saved_files.items())[:5], 1):
                print(f"{i}. {player}")
                print(f"   -> {filepath}")

            if len(saved_files) > 5:
                print(f"\n... and {len(saved_files) - 5} more players")

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise


if __name__ == "__main__":
    main()
