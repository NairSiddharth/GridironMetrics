import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from constants import TEAM_CODES

START_YEAR = 2000  # change if you want earlier data
records = []

for code, team_name in TEAM_CODES.items():
    url = f"https://www.pro-football-reference.com/teams/{code}/"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    
    # The main "Team index" table contains year-by-year results
    table = soup.find("table", id="team_index")
    if not table:
        print(f"No table found for {team_name}")
        continue
    
    for row in table.select("tbody tr"):
        year_cell = row.find("th", {"data-stat": "year_id"})
        if not year_cell or not year_cell.text.strip().isdigit():
            continue
        year = int(year_cell.text.strip())
        if year < START_YEAR:
            continue

        coach_cell = row.find("td", {"data-stat": "coach"})
        w_cell = row.find("td", {"data-stat": "wins"})
        l_cell = row.find("td", {"data-stat": "losses"})
        t_cell = row.find("td", {"data-stat": "ties"})

        coach = coach_cell.text.strip() if coach_cell else None
        w = w_cell.text.strip() if w_cell else None
        l = l_cell.text.strip() if l_cell else None
        t = t_cell.text.strip() if t_cell else None

        records.append({
            "Team": team_name,
            "Abbr": code.upper(),
            "Year": year,
            "Coach": coach,
            "Wins": w,
            "Losses": l,
            "Ties": t,
        })

    print(f"{team_name} done.")
    time.sleep(0.5)  # Avoid getting blocked

df = pd.DataFrame(records)
df.sort_values(["Year", "Team"], inplace=True)
df.to_csv("cache/nfl_coaches.csv", index=False)

print(f"\nFinished! {len(df)} rows written to nfl_coaches.csv")
