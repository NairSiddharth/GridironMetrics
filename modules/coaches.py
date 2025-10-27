import requests
from bs4 import BeautifulSoup, Comment
import polars as pl
import time
from pathlib import Path
import sys
from urllib.parse import urlparse

# polite request headers to avoid basic blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                  " Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}
from constants import TEAM_CODES, START_YEAR, END_YEAR, RATE_LIMIT_STRICT, RATE_LIMIT_OTHER, JAIL_DURATION_SECONDS
import json

# Root cache directory
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

for code, team_name in TEAM_CODES.items():
    team_cache_dir = CACHE_DIR / code
    team_cache_dir.mkdir(exist_ok=True)

    records = []

    session = requests.Session()
    session.headers.update(HEADERS)

    # Simple per-domain rate limiting
    last_request_time = {}

    def _domain_for(url):
        return urlparse(url).netloc.lower()

    def _is_strict_domain(domain):
        # Treat Pro-Football-Reference / FBref / Stathead / Sports-Reference as strict
        strict_keywords = ["pro-football-reference.com", "fbref.com", "stathead.com", "sports-reference.com"]
        return any(k in domain for k in strict_keywords)

    def throttle(url):
        dom = _domain_for(url)
        # jail check
        if is_jailed(dom):
            jailed_until = get_jail_info().get(dom)
            print(f"Domain {dom} is jailed until {time.ctime(jailed_until)}; skipping request")
            return False
        now = time.time()
        # rules: strict domains -> RATE_LIMIT_STRICT per minute, others -> RATE_LIMIT_OTHER per minute
        interval = 60.0 / RATE_LIMIT_STRICT if _is_strict_domain(dom) else 60.0 / RATE_LIMIT_OTHER
        last = last_request_time.get(dom, 0)
        elapsed = now - last
        if elapsed < interval:
            to_sleep = interval - elapsed
            # small log
            print(f"Throttling {dom}: sleeping {to_sleep:.2f}s to respect rate limit")
            time.sleep(to_sleep)
        last_request_time[dom] = time.time()
        return True


    # jail persistence helpers
    JAIL_FILE = CACHE_DIR / "jail.json"

    def get_jail_info():
        try:
            if JAIL_FILE.exists():
                with open(JAIL_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # ensure numeric timestamps
                    return {k: float(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def save_jail_info(jail_dict):
        try:
            with open(JAIL_FILE, "w", encoding="utf-8") as f:
                json.dump(jail_dict, f)
        except Exception as e:
            print(f"Failed to save jail file: {e}")

    def is_jailed(domain):
        jail = get_jail_info()
        now = time.time()
        if domain in jail:
            until = float(jail[domain])
            if now < until:
                return True
            else:
                # expired -> remove and save
                jail.pop(domain, None)
                save_jail_info(jail)
        return False

    def apply_jail(domain, duration_seconds=JAIL_DURATION_SECONDS):
        jail = get_jail_info()
        jail[domain] = time.time() + float(duration_seconds)
        save_jail_info(jail)

    for year in range(START_YEAR, END_YEAR + 1):
        url = f"https://www.pro-football-reference.com/teams/{code}/{year}.htm"
        # throttle per-domain before request
        dom = _domain_for(url)
        ok = throttle(url)
        if not ok:
            # domain is jailed and throttle returned False
            continue
        # simple retry/backoff for 429s or transient errors
        retries = 3
        backoff = 2
        r = None
        for attempt in range(retries):
            try:
                r = session.get(url, timeout=20)
            except requests.RequestException as e:
                print(f"Request error fetching {team_name} {year}: {e} (attempt {attempt+1})")
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code == 429:
                print(f"Got 429 for {team_name} {year}, backing off (attempt {attempt+1})")
                time.sleep(backoff)
                backoff *= 2
                continue
            break
        if r is None:
            print(f"Failed to fetch {team_name} {year} after retries")
            continue
        # If we get a 403, apply local jail for this domain
        if r.status_code == 403:
            print(f"Received 403 for {team_name} {year}; applying local jail for domain {dom}")
            try:
                apply_jail(dom)
            except Exception as e:
                print(f"Failed to apply jail: {e}")
            continue
        if r.status_code != 200:
            print(f"Failed to fetch {team_name} {year} (status {r.status_code})")
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        # Pro-Football-Reference often places large tables inside HTML comments.
        # Try to find the table normally, and if not present, look through
        # HTML comments and parse them to find the commented table.
        table = soup.find("table", id="games")
        if not table:
            # search commented sections for the table markup
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for c in comments:
                if 'id="games"' in c or "id='games'" in c:
                    try:
                        inner = BeautifulSoup(c, "html.parser")
                        table = inner.find("table", id="games")
                        if table:
                            break
                    except Exception:
                        # fallback: continue scanning other comments
                        continue
        if not table:
            print(f"No games table found for {team_name} {year}")
            continue

        for row in table.select("tbody tr"):
            coach_cell = row.find("td", {"data-stat": "coach"})
            if not coach_cell:
                continue
            coach_name = coach_cell.text.strip()
            if not coach_name:
                continue

            # Wins, Losses, Ties for this coach in this game
            w_cell = row.find("td", {"data-stat": "game_result"})
            if not w_cell or not w_cell.text.strip():
                continue
            result = w_cell.text.strip()
            wins = 1 if result.startswith("W") else 0
            losses = 1 if result.startswith("L") else 0
            ties = 1 if result.startswith("T") else 0
            total_games = wins + losses + ties

            records.append({
                "Year": year,
                "Team": team_name,
                "Abbr": code.upper(),
                "Coach": coach_name,
                "Wins": wins,
                "Losses": losses,
                "Ties": ties,
                "Games_Coached": total_games
            })

        time.sleep(0.5)  # polite scraping

    if records:
        df = pl.DataFrame(records)
        # Aggregate per coach per season
        df_agg = df.groupby(["Year", "Team", "Abbr", "Coach"], maintain_order=True).agg([
            pl.sum("Wins").alias("Wins"),
            pl.sum("Losses").alias("Losses"),
            pl.sum("Ties").alias("Ties"),
            pl.sum("Games_Coached").alias("Games_Coached")
        ])
        df_agg.write_csv(team_cache_dir / f"{code}_coaches.csv")
        print(f"{team_name} CSV written with {df_agg.height} rows.")
    else:
        print(f"No data found for {team_name}")
