import requests
from bs4 import BeautifulSoup, Comment
import json
from pathlib import Path
import time
import sys
# ensure we can import the project's modules package by adding repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from modules.constants import JAIL_DURATION_SECONDS

URLS = [
    "https://www.pro-football-reference.com/teams/oti/2025.htm",
    "https://www.pro-football-reference.com/teams/oti/2025/gamelog/",
]

def find_games_table(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table", id="games")
    if table:
        return table, 'normal'

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for c in comments:
        if 'id="games"' in c or "id='games'" in c:
            inner = BeautifulSoup(c, "html.parser")
            table = inner.find("table", id="games")
            if table:
                return table, 'comment'
    return None, None


def main():
    import time
    from urllib.parse import urlparse

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                      " Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.pro-football-reference.com/",
    })

    last_request = {}
    # jail file (shared with main scraper)
    CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
    JAIL_FILE = CACHE_DIR / "jail.json"

    def get_jail_info():
        try:
            if JAIL_FILE.exists():
                with open(JAIL_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {k: float(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def save_jail_info(jail_dict):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(JAIL_FILE, "w", encoding="utf-8") as f:
                json.dump(jail_dict, f)
        except Exception as e:
            print("Failed to save jail info:", e)

    def is_jailed(domain):
        jail = get_jail_info()
        now = time.time()
        if domain in jail:
            until = float(jail[domain])
            if now < until:
                return True, until
            else:
                jail.pop(domain, None)
                save_jail_info(jail)
        return False, None

    def apply_jail(domain, duration_seconds=JAIL_DURATION_SECONDS):
        jail = get_jail_info()
        jail[domain] = time.time() + float(duration_seconds)
        save_jail_info(jail)

    def domain(url):
        return urlparse(url).netloc.lower()

    def is_strict(dom):
        kws = ["pro-football-reference.com", "fbref.com", "stathead.com", "sports-reference.com"]
        return any(k in dom for k in kws)

    def throttle(u):
        d = domain(u)
        now = time.time()
        interval = 60.0/10.0 if is_strict(d) else 60.0/20.0
        last = last_request.get(d, 0)
        if now - last < interval:
            to_sleep = interval - (now - last)
            print(f"Test throttle: sleeping {to_sleep:.2f}s for {d}")
            time.sleep(to_sleep)
        last_request[d] = time.time()

    rows = []
    for URL in URLS:
        print(f"\nFetching {URL}")
        # jail check
        dom = domain(URL)
        jailed, until = is_jailed(dom)
        if jailed:
            print(f"Domain {dom} is jailed until {time.ctime(until)}; skipping test request")
            continue
        throttle(URL)
        try:
            r = session.get(URL, timeout=20)
        except requests.RequestException as e:
            print("Request error:", e)
            continue
        print("HTTP", r.status_code)
        if r.status_code == 403:
            print(f"Test received 403 from {dom}; applying local jail")
            apply_jail(dom)
            continue
        table, where = find_games_table(r.text)
        if not table:
            print("No games table found (normal or commented).")
            continue
        print(f"Found table (from {where})")
        rows = table.select("tbody tr")
        print(f"Total rows in tbody: {len(rows)}")
    sample = []
    for i, row in enumerate(rows[:10]):
        coach_cell = row.find("td", {"data-stat": "coach"})
        result_cell = row.find("td", {"data-stat": "game_result"})
        coach = coach_cell.text.strip() if coach_cell else None
        result = result_cell.text.strip() if result_cell else None
        sample.append((i+1, coach, result))
    for s in sample:
        print(s)

if __name__ == '__main__':
    main()
